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
static struct COOMatrix* B;
static struct CSRMatrix* A;
static val_dt* x;
static val_dt* y;
static struct partition_info_t *part_info;

/**
 * @brief Specific information per DPU
 */
struct dpu_info_t {
    uint32_t rows_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
    uint32_t nnz;
    uint32_t nnz_pad;
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
 * @brief compute output in the host CPU
 */
static void spmv_host(val_dt* y, struct CSRMatrix *A, val_dt* x) {

    for(unsigned int rowIndx = 0; rowIndx < A->nrows; ++rowIndx) {
        val_dt sum = 0;
        for(unsigned int i = A->rowptr[rowIndx]; i < A->rowptr[rowIndx + 1]; ++i) {
            unsigned int colIndx = A->colind[i];
            val_dt value = A->values[i];
            sum += x[colIndx] * value;
        }
        y[rowIndx] = sum;
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

    unsigned int i;

    // Initialize input data 
    B = readCOOMatrix(p.fileName);
    A = coo2csr(B);
    freeCOOMatrix(B);

    // Initialize partition data
    part_info = partition_init(nr_of_dpus, NR_TASKLETS);

    // Load-balance row across DPUs
    partition_by_row(A, part_info, nr_of_dpus);

    // Allocate input vector
    x = (val_dt *) malloc((A->ncols) * sizeof(val_dt)); 

    // Initialize input vector with arbitrary data
    init_vector(x, A->ncols);
    
    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_rows_per_dpu = 0;
    uint64_t max_nnz_ind_per_dpu = 0;
    uint64_t max_nnz_val_per_dpu = 0;
    uint64_t max_rows_per_tasklet = 0;

    // Timer for measurements
    Timer timer;

    i = 0;
    // Find padding for rows and non-zero elements needed for CPU-DPU transfers
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t rows_per_dpu = part_info->row_split[i+1] - part_info->row_split[i];
        uint32_t prev_rows_dpu = part_info->row_split[i];

        // Pad data to be transfered for rows
        uint32_t rows_per_dpu_pad = rows_per_dpu + 1;
        if (rows_per_dpu_pad % (8 / byte_dt) != 0)
            rows_per_dpu_pad += ((8 / byte_dt) - (rows_per_dpu_pad % (8 / byte_dt)));
#if INT64 || FP64
        if (rows_per_dpu_pad % 2 == 1)
            rows_per_dpu_pad++;
#endif
        if (rows_per_dpu_pad > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu_pad;

        // Pad data to be transfered for nnzs
        unsigned int nnz, nnz_ind_pad, nnz_val_pad;
        nnz = A->rowptr[rows_per_dpu + prev_rows_dpu] - A->rowptr[prev_rows_dpu];
        if (nnz % 2 != 0)
            nnz_ind_pad = nnz + 1;
        else
            nnz_ind_pad = nnz;
        if (nnz % (8 / byte_dt) != 0)
            nnz_val_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
        else
            nnz_val_pad = nnz;

#if INT64 || FP64
        if (nnz_ind_pad % 2 == 1)
            nnz_ind_pad++;
        if (nnz_val_pad % 2 == 1)
            nnz_val_pad++;
#endif
        if (nnz_ind_pad > max_nnz_ind_per_dpu)
            max_nnz_ind_per_dpu = nnz_ind_pad;
        if (nnz_val_pad > max_nnz_val_per_dpu)
            max_nnz_val_per_dpu = nnz_val_pad;

        // Keep information per DPU
        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].nnz = nnz;

        // Find input arguments per DPU
        input_args[i].nrows = rows_per_dpu;
        input_args[i].tcols = A->ncols; 

#if BLNC_TSKLT_ROW
        // Load-balance rows across tasklets 
        partition_tsklt_by_row(part_info, rows_per_dpu, NR_TASKLETS);
#else
        // Load-balance nnz across tasklets 
        partition_tsklt_by_nnz(A, part_info, rows_per_dpu, nnz, prev_rows_dpu, NR_TASKLETS);
#endif
        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            // Find input arguments per DPU
            input_args[i].start_row[t] = part_info->row_split_tasklet[t]; 
            input_args[i].rows_per_tasklet[t] = part_info->row_split_tasklet[t+1] - part_info->row_split_tasklet[t];

            if (input_args[i].rows_per_tasklet[t] > max_rows_per_tasklet)
                max_rows_per_tasklet = input_args[i].rows_per_tasklet[t];
        }
    }

    // Initializations for parallel transfers with padding needed
    if (max_rows_per_dpu % 2 != 0)
        max_rows_per_dpu++;
    if (max_nnz_ind_per_dpu % 2 != 0)
        max_nnz_ind_per_dpu++;
    if (max_nnz_val_per_dpu % (8 / byte_dt) != 0)
        max_nnz_val_per_dpu += ((8 / byte_dt) - (max_nnz_val_per_dpu % (8 / byte_dt)));
    if (max_rows_per_tasklet % (8 / byte_dt) != 0)
        max_rows_per_tasklet += ((8 / byte_dt) - (max_rows_per_tasklet % (8 / byte_dt)));

    // Re-allocations for padding needed
    A->rowptr = (uint32_t *) realloc(A->rowptr, (max_rows_per_dpu * nr_of_dpus * sizeof(uint32_t)));
    A->colind = (uint32_t *) realloc(A->colind, (max_nnz_ind_per_dpu * nr_of_dpus * sizeof(uint32_t)));
    A->values = (val_dt *) realloc(A->values, (max_nnz_val_per_dpu * nr_of_dpus * sizeof(val_dt)));
    y = (val_dt *) malloc((uint64_t) ((uint64_t) nr_of_dpus * (uint64_t) max_rows_per_dpu) * (uint64_t) sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_rows_per_dpu) * sizeof(uint32_t)) + (max_nnz_ind_per_dpu * sizeof(uint32_t)) + (max_nnz_val_per_dpu * sizeof(val_dt)) + (A->ncols * sizeof(val_dt)) + (max_rows_per_dpu * sizeof(val_dt));
    assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");



    // Copy input arguments to DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        input_args[i].max_rows = max_rows_per_dpu; 
        input_args[i].max_nnz_ind = max_nnz_ind_per_dpu; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));


    // Copy input matrix to DPUs
    startTimer(&timer, 0);

    // Copy Rowptr 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->rowptr + dpu_info[i].prev_rows_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (max_rows_per_dpu * sizeof(val_dt) + A->ncols * sizeof(val_dt)), max_rows_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));

    // Copy Colind
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->colind + A->rowptr[dpu_info[i].prev_rows_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + A->ncols * sizeof(val_dt) + max_rows_per_dpu * sizeof(uint32_t), max_nnz_ind_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));

    // Copy Values
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->values + A->rowptr[dpu_info[i].prev_rows_dpu]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + A->ncols * sizeof(val_dt) + max_rows_per_dpu * sizeof(uint32_t) + max_nnz_ind_per_dpu * sizeof(uint32_t), max_nnz_val_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);



    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, x));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), A->ncols * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 1);



    // Run kernel on DPUs
    startTimer(&timer, 2);
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    stopTimer(&timer, 2);

#if LOG
    // Display DPU Logs (default: disabled)
    DPU_FOREACH(dpu_set, dpu) {
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif


    // Retrieve results for output vector from DPUs
    startTimer(&timer, 3);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_rows_per_dpu)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);


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
    printf("\n\n");


#if CHECK_CORR
    // Check output
    val_dt *y_host = (val_dt *) malloc((A->nrows) * sizeof(val_dt)); 
    spmv_host(y_host, A, x); 

    bool status = true;
    i = 0;
    unsigned int n,j,t;
    for (n = 0; n < nr_of_dpus; n++) {
            unsigned int cur_rows = dpu_info[n].rows_per_dpu;
            for (j = 0; j < cur_rows; j++) {
                if(y_host[i] != y[n * max_rows_per_dpu + j]) {
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
    freeCSRMatrix(A);
    free(x);
    free(y);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
