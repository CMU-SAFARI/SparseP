/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
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
#include <math.h>
#include <omp.h>

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
static struct DBCOOMatrix* A;
static struct DBCSRMatrix* B;
static struct DCSRMatrix* C;
static struct COOMatrix* D;
static val_dt* x;
static val_dt* y;
static struct partition_info_t *part_info;


/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t block_rows_per_dpu;
    uint32_t prev_block_rows_dpu;
    uint32_t block_start;
    uint32_t blocks;
    uint32_t blocks_pad;
    uint32_t merge;
};

struct dpu_info_t *dpu_info;


/**
 * @brief find the dpus_per_vert_partition
 * @param factor n to create partitions
 * @param vertical_partitions 
 * @param/return horz_partitions 
 */
void find_partitions(uint32_t n, uint32_t *horz_partitions, uint32_t vert_partitions) {
    uint32_t dpus_per_vert_partition = n / vert_partitions;
    *horz_partitions = dpus_per_vert_partition;
}

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
void spmv_host(val_dt *y, struct DBCOOMatrix *dbcooMtx, val_dt *x) {

    uint64_t total_blocks = 0;
    for (uint32_t r = 0; r < dbcooMtx->horz_partitions; r++) {
        for (uint32_t c = 0; c < dbcooMtx->vert_partitions; c++) {
            uint32_t partition = r * dbcooMtx->vert_partitions + c;
            for(uint64_t n=0; n<dbcooMtx->blocks_per_partition[partition]; n++) {
                uint64_t i = dbcooMtx->bind[total_blocks + n].rowind;
                uint64_t j = dbcooMtx->bind[total_blocks + n].colind;
                for(uint64_t blr=0; blr<dbcooMtx->row_block_size; blr++){
                    val_dt acc = 0;
                    for(uint64_t blc=0; blc<dbcooMtx->col_block_size; blc++) {
                        acc += dbcooMtx->bval[total_blocks * dbcooMtx->row_block_size * dbcooMtx->col_block_size + n * dbcooMtx->col_block_size * dbcooMtx->row_block_size + blr * dbcooMtx->col_block_size + blc] * x[c * dbcooMtx->tile_width + j * dbcooMtx->col_block_size + blc];
                    }
                    y[r * dbcooMtx->tile_height + i * dbcooMtx->row_block_size + blr] += acc;
                }
            }
            total_blocks += dbcooMtx->blocks_per_partition[partition];
        }
    }
}


/**
 * @brief main of the host application
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

    unsigned int i;

    // Initialize input data 
    D = readCOOMatrix(p.fileName);
    sortCOOMatrix(D);
    uint32_t horz_partitions = 0;
    uint32_t vert_partitions = p.vert_partitions;
    find_partitions(nr_of_dpus, &horz_partitions, p.vert_partitions);
    printf("[INFO] %dx%d Matrix Partitioning\n\n", horz_partitions, vert_partitions);
    C = coo2dcsr(D, horz_partitions, vert_partitions);
    freeCOOMatrix(D);
    B = dcsr2dbcsr(C, p.row_blsize, p.col_blsize);
    sortDBCSRMatrix(B);
    countNNZperBlockDBCSRMatrix(B);
    freeDCSRMatrix(C);
    A = dbcsr2dbcoo(B);
    freeDBCSRMatrix(B);

    // Initialize partition data
    part_info = partition_init(nr_of_dpus, NR_TASKLETS);

    // Initialize help data - Padding needed
    uint32_t ncols_pad = A->vert_partitions * A->tile_width + A->col_block_size;
    uint32_t tile_width_pad = A->num_block_cols * A->col_block_size;
    uint32_t nrows_pad = A->horz_partitions * A->tile_height + A->row_block_size;
    if (ncols_pad % (8 / byte_dt) != 0)
        ncols_pad = ncols_pad + ((8 / byte_dt) - (ncols_pad % (8 / byte_dt)));
    if (tile_width_pad % (8 / byte_dt) != 0)
        tile_width_pad = tile_width_pad + ((8 / byte_dt) - (tile_width_pad % (8 / byte_dt)));
#if INT8
    if (tile_width_pad % 2 != 0)
        tile_width_pad++;
#endif
    if (nrows_pad % (8 / byte_dt) != 0)
        nrows_pad = nrows_pad + ((8 / byte_dt) - (nrows_pad % (8 / byte_dt)));

    // Allocate input vector
    x = (val_dt *) malloc(ncols_pad * sizeof(val_dt)); 

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

    i = 0;
    uint32_t total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Find padding for block rows and non-zero elements needed for CPU-DPU transfers
        uint64_t block_rows_per_dpu = A->num_block_rows;
        uint64_t prev_block_rows_dpu = 0;

        if (block_rows_per_dpu > max_block_rows_per_dpu)
            max_block_rows_per_dpu = block_rows_per_dpu;

        unsigned int blocks;
        blocks = A->blocks_per_partition[i];

        if (blocks > max_blocks_per_dpu)
            max_blocks_per_dpu = blocks;

        // Keep information per DPU
        dpu_info[i].block_rows_per_dpu = block_rows_per_dpu;
        dpu_info[i].prev_block_rows_dpu = prev_block_rows_dpu;
        dpu_info[i].blocks = blocks;

        // Find input arguments per DPU
        input_args[i].block_rows = block_rows_per_dpu;
        input_args[i].start_block_row = 0;
        input_args[i].tcols = tile_width_pad; 
        input_args[i].row_block_size = A->row_block_size; 
        input_args[i].col_block_size = A->col_block_size; 
        //input_args[i].blocks = blocks; 

#if BLNC_TSKLT_BLOCK
        // Load-balance blocks across tasklets 
        partition_tsklt_by_block(A, part_info, i, NR_TASKLETS, nr_of_dpus, total_blocks);
#else
        // Load-balance nnzs across tasklets 
        partition_tsklt_by_nnz(A, part_info, i, NR_TASKLETS, nr_of_dpus, total_blocks);
#endif

        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            // Find input arguments per DPU
            input_args[i].start_block[t] = part_info->block_split_tasklet[i * (NR_TASKLETS+2) + t]; 
            input_args[i].blocks_per_tasklet[t] = part_info->block_split_tasklet[i * (NR_TASKLETS+2) + (t+1)] - part_info->block_split_tasklet[i * (NR_TASKLETS+2) + t];
        }

        total_blocks += A->blocks_per_partition[i];
    }


    // Initialization for parallel transfers 
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
    total_bytes = (max_blocks_per_dpu * sizeof(struct bind_t)) + (max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt)) + (tile_width_pad * sizeof(val_dt)) + (max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt));
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

    // Copy Browind + Bcolind
    i = 0;
    total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bind + total_blocks));
        total_blocks += A->blocks_per_partition[i];
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + tile_width_pad * sizeof(val_dt), max_blocks_per_dpu * sizeof(struct bind_t), DPU_XFER_DEFAULT));

    // Copy Values
    i = 0;
    total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bval + ((uint64_t) total_blocks * A->row_block_size * A->col_block_size)));
        total_blocks += A->blocks_per_partition[i];
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + tile_width_pad * sizeof(val_dt) + max_blocks_per_dpu * sizeof(struct bind_t), max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);


    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i % A->vert_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + tile_vert_indx * A->tile_width));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt), tile_width_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
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
    uint32_t r, c, t;
#pragma omp parallel for num_threads(p.nthreads) shared(A, y, max_block_rows_per_dpu) private(r,c,t) collapse(2) 
    for (r = 0; r < A->horz_partitions; r++) {
        for (t = 0; t < A->tile_height; t++) {
            for (c = 1; c < A->vert_partitions; c++) {
                y[r * A->vert_partitions * max_block_rows_per_dpu * A->row_block_size + t] += y[r * A->vert_partitions * max_block_rows_per_dpu * A->row_block_size + c * max_block_rows_per_dpu * A->row_block_size + t];
            }
        }
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
    val_dt *y_host = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
    spmv_host(y_host, A, x); 

    bool status = true;
    i = 0;
    for (uint32_t r = 0; r < A->horz_partitions; r++) {
        for (uint32_t t = 0; t < A->tile_height; t++) {
            if((r * A->tile_height + t < A->nrows) && y_host[i] != y[r * A->vert_partitions * max_block_rows_per_dpu * A->row_block_size + t]) {
                status = false;
            }
            i++;
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
    freeDBCOOMatrix(A);
    free(x);
    free(y);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));
    return 0;

}
