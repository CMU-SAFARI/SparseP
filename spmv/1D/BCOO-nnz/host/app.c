/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
#include "../support/partition.h"
#include "../support/timer.h"
#include "../support/utils.h"

// Define the DPU Binary path as DPU_BINARY here.
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/spmv_dpu"
#endif

#define DPU_CAPACITY (64 << 20) // A DPU's capacity is 64 MB

/*
 * Main Structures:
 * 1. Matrices
 * 2. Input vector
 * 3. Output vector
 * 4. Help structures for data partitioning
 */
static struct COOMatrix* D;
static struct CSRMatrix* C;
static struct BCSRMatrix* B;
static struct BCOOMatrix* A;
static val_dt* x;
static val_dt* y;
static struct partition_info_t *part_info;

/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t block_rows_per_dpu;
    uint32_t start_block_row_dpu;
    uint32_t start_block_dpu;
    uint32_t blocks;
    uint32_t blocks_pad;
    uint32_t merge;
};
struct dpu_info_t *dpu_info;



/**
 * @brief initialize input vector 
 * @param pointer to input vector and vector size
 */
void init_vector(val_dt* vec, uint32_t size) {
    for(unsigned int i = 0; i < size; ++i) {
        vec[i] = (val_dt) (i%4+1);
    }
}

/**
 * @brief compute output in the host
 */
void spmv_host(val_dt *y, struct BCOOMatrix *bcooMtx, val_dt *x) {
    for(uint64_t n=0; n<bcooMtx->num_blocks; n++) {
        uint64_t i = bcooMtx->bind[n].rowind;
        uint64_t j = bcooMtx->bind[n].colind;
        for(uint64_t r=0; r<bcooMtx->row_block_size; r++){
            val_dt acc = 0;
            for(uint64_t c=0; c<bcooMtx->col_block_size; c++) {
                if ((i * bcooMtx->row_block_size + r < bcooMtx->nrows) && (j * bcooMtx->col_block_size + c < bcooMtx->ncols)) {
                    acc += bcooMtx->bval[n * bcooMtx->col_block_size * bcooMtx->row_block_size + r * bcooMtx->col_block_size + c] * x[j * bcooMtx->col_block_size + c];
                }
            }
            y[i * bcooMtx->row_block_size + r] += acc;
        }
    }
}

/**
 * @brief main of the host application.
 */
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("[INFO] Allocated %d DPU(s)\n", nr_of_dpus);
    printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);

    uint64_t i;

    // Initialize input data 
    D = readCOOMatrix(p.fileName);
    C = coo2csr(D);
    B = csr2bcsr(C, p.row_blsize, p.col_blsize);
    freeCOOMatrix(D);
    freeCSRMatrix(C);
    sortBCSRMatrix(B); 
    countNNZperBlockBCSRMatrix(B);
    A = bcsr2bcoo(B);
    freeBCSRMatrix(B);

    // Initialize partition data
    part_info = partition_init(nr_of_dpus, NR_TASKLETS);
 
    // Load-balance nnz across DPUs
    partition_by_nnz(A, part_info, nr_of_dpus);

    // Initialize help data with padding if needed
    uint64_t ncols_pad = A->num_block_cols * A->col_block_size;
    if (ncols_pad % (8 / byte_dt) != 0)
        ncols_pad += ((8 / byte_dt) - (A->ncols % (8 / byte_dt)));
#if INT8
    if (ncols_pad % 8 != 0)
        ncols_pad += (8 - (ncols_pad % 8));
#endif
    uint64_t nrows_pad = A->num_block_rows * A->row_block_size;
    if (nrows_pad % (8 / byte_dt) != 0)
        nrows_pad += ((8 / byte_dt) - (A->nrows % (8 / byte_dt)));

    // Allocate input vector
    x = (val_dt *) malloc((ncols_pad) * sizeof(val_dt)); 

    // Initialize input vector with arbitrary data
    init_vector(x, ncols_pad);

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));

    // Max limits for parallel transfers
    uint64_t max_block_rows_per_dpu = 0;
    uint64_t max_blocks_per_dpu = 0;

    // Timer for measurements
    Timer timer;

    uint32_t prev_block_row = 0;
    i = 0;
    // Find padding for block-rows and non-zero elements needed for CPU-DPU transfers
    DPU_FOREACH(dpu_set, dpu, i) {
        unsigned int blocks, blocks_pad;
        blocks = (part_info->block_split[i+1] - part_info->block_split[i]);
        if (blocks % 2 != 0) // browind, bcolind 
            blocks_pad = blocks + 1;
        else
            blocks_pad = blocks;

        if (blocks_pad > max_blocks_per_dpu)
            max_blocks_per_dpu = blocks_pad;

        unsigned int block_rows_dpu = A->bind[part_info->block_split[i+1] - 1].rowind - prev_block_row + 1;
        if (block_rows_dpu > max_block_rows_per_dpu)
            max_block_rows_per_dpu = block_rows_dpu;

        // Keep information per DPU
        dpu_info[i].block_rows_per_dpu = block_rows_dpu; 
        dpu_info[i].start_block_row_dpu = prev_block_row; 
        dpu_info[i].start_block_dpu = part_info->block_split[i];
        dpu_info[i].blocks = blocks;
        dpu_info[i].blocks_pad = blocks_pad;
        if (i == 0)
            dpu_info[i].merge = 0;
        else if (A->bind[part_info->block_split[i]].rowind == A->bind[part_info->block_split[i]-1].rowind)
            dpu_info[i].merge = 1;
        else
            dpu_info[i].merge = 0;

        // Find input arguments per DPU
        input_args[i].block_rows = block_rows_dpu;
        input_args[i].start_block_row = prev_block_row;
        input_args[i].tcols = ncols_pad; 
        input_args[i].row_block_size = A->row_block_size; 
        input_args[i].col_block_size = A->col_block_size; 

        // Compute prev_block_row (previous block row) for the next DPU 
        if (i != nr_of_dpus) {
            if (A->bind[part_info->block_split[i+1]].rowind == A->bind[part_info->block_split[i+1] - 1].rowind) 
                prev_block_row = A->bind[part_info->block_split[i+1] - 1].rowind;
            else
                prev_block_row = A->bind[part_info->block_split[i+1] - 1].rowind + 1;
        } else {
            prev_block_row = A->num_block_rows;
        }

#if BLNC_TSKLT_BLOCK
        // load-balance rows across tasklets 
        partition_tsklt_by_block(A, part_info, i, NR_TASKLETS, nr_of_dpus);
#else
        // load-balance nnz across tasklets 
        partition_tsklt_by_nnz(A, part_info, i, NR_TASKLETS, nr_of_dpus);
#endif

        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            // Find input arguments per DPU
            input_args[i].start_block[t] = part_info->block_split_tasklet[i * (NR_TASKLETS+2) + t]; 
            input_args[i].blocks_per_tasklet[t] = part_info->block_split_tasklet[i * (NR_TASKLETS+2) + (t+1)] - part_info->block_split_tasklet[i * (NR_TASKLETS+2) + t];
        }

    }

    // Initializations for parallel transfers with padding needed
#if INT8
    if (max_block_rows_per_dpu % 2 != 0)
        max_block_rows_per_dpu++;
#endif
    if (max_blocks_per_dpu % 2 != 0)
        max_blocks_per_dpu++;

    // Re-allocations for padding needed
    A->bind = (struct bind_t *) realloc(A->bind, (max_blocks_per_dpu * nr_of_dpus * sizeof(struct bind_t)));
    A->bval = (val_dt *) realloc(A->bval, (max_blocks_per_dpu * A->row_block_size * A->col_block_size * nr_of_dpus * sizeof(val_dt)));
    y = (val_dt *) calloc((uint64_t) ((uint64_t) nr_of_dpus * (uint64_t) max_block_rows_per_dpu * A->row_block_size), sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_blocks_per_dpu) * sizeof(struct bind_t)) + (max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt)) + (ncols_pad * sizeof(val_dt)) + (max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt));
    assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");


    // Copy input arguments to DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        input_args[i].max_block_rows = max_block_rows_per_dpu; 
        input_args[i].max_blocks = max_blocks_per_dpu; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input matrix to DPUs
    startTimer(&timer, 0);

    // Copy Rowind + Colind
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bind + part_info->block_split[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + ncols_pad * sizeof(val_dt)), max_blocks_per_dpu * sizeof(struct bind_t), DPU_XFER_DEFAULT));

    // Copy Bvalues
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bval + (((uint64_t) part_info->block_split[i]) * A->row_block_size * A->col_block_size)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + ncols_pad * sizeof(val_dt) + max_blocks_per_dpu * sizeof(struct bind_t), max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);


    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, x));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt), ncols_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 1);

    // Run kernel on DPUs
    startTimer(&timer, 2);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    stopTimer(&timer, 2);

#if LOG
    // Display DPU Log (default: disabled)
    DPU_FOREACH(dpu_set, dpu) {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

    // Retrieve results for output vector from DPUs
    startTimer(&timer, 3);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_block_rows_per_dpu * A->row_block_size)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);


    // Merge partial results to the host CPU
    startTimer(&timer, 4);
    uint64_t t, n = 0;
    while (n < nr_of_dpus - 1) {
        uint64_t actual_block_rows = dpu_info[n].block_rows_per_dpu - 1;
        if (dpu_info[n].start_block_row_dpu + actual_block_rows == dpu_info[n+1].start_block_row_dpu) {
            t = n;
            // Merge multiple y[i] elements computed in different DPUs
            while (dpu_info[n].start_block_row_dpu + actual_block_rows == dpu_info[t+1].start_block_row_dpu) {
                for(uint64_t r = 0; r < A->row_block_size; r++) {
                    y[n * max_block_rows_per_dpu * A->row_block_size + actual_block_rows * A->row_block_size + r] += y[(t+1) * max_block_rows_per_dpu * A->row_block_size + r];
                }
                t++;
            }
        }
        n++;
    }
    stopTimer(&timer, 4);


    // Print timing results
    printf("\n");
    printf("Load Matrix ");
    printTimer(&timer, 0);
    printf("Load Input Vector ");
    printTimer(&timer, 1);
    printf("Kernel ");
    printTimer(&timer, 2);
    printf("Retrieve Output Vector ");
    printTimer(&timer, 3);
    printf("Merge Partial Results ");
    printTimer(&timer, 4);
    printf("\n\n");


#if CHECK_CORR
    // Check output
    val_dt *y_host = (val_dt *) calloc((nrows_pad), sizeof(val_dt)); 
    spmv_host(y_host, A, x); 

    bool status = true;
    i = 0;
    unsigned int j,r;
    for (n = 0; n < nr_of_dpus; n++) {
        uint64_t actual_block_rows = dpu_info[n].block_rows_per_dpu;
        for (j = 0; j < actual_block_rows; j++) {
            if(j == 0 && dpu_info[n].merge == 1) {
                continue; 
            }
            for (r = 0; r < A->row_block_size; r++) {
                if(y_host[i] != y[n * max_block_rows_per_dpu * A->row_block_size + j * A->row_block_size + r] && i < A->nrows) {
                    status = false;
                }
                i++;
            }
        }
    }
    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    free(y_host);
#endif




    // Deallocation
    freeBCOOMatrix(A);
    free(x);
    free(y);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
