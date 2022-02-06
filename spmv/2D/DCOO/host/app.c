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
#include <omp.h>

#include "../support/common.h"
#include "../support/matrix.h"
#include "../support/params.h"
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
 */
static struct DCOOMatrix* A;
static struct COOMatrix* B;
static val_dt* x;
static val_dt* y;


/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t rows_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
    uint32_t prev_nnz_dpu;
    uint32_t nnz;
    uint32_t nnz_pad;
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
static void spmv_host(val_dt* y, struct DCOOMatrix *A, val_dt* x) {

    uint64_t total_nnzs = 0;
    for (uint32_t r = 0; r < A->horz_partitions; r++) {
        for (uint32_t c = 0; c < A->vert_partitions; c++) {
            uint32_t p = r * A->vert_partitions + c;
            for(uint32_t n = 0; n < A->nnzs_per_partition[p]; n++) {
                uint32_t rowIndx = A->nnzs[total_nnzs].rowind;    
                uint32_t colIndx = A->nnzs[total_nnzs].colind;    
                val_dt value = A->nnzs[total_nnzs++].val;    
                y[r * A->tile_height + rowIndx] += (value * x[c * A->tile_width + colIndx]); 

            }
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
    B = readCOOMatrix(p.fileName);
    sortCOOMatrix(B);
    uint32_t horz_partitions = 0;
    uint32_t vert_partitions = p.vert_partitions; 
    find_partitions(nr_of_dpus, &horz_partitions, p.vert_partitions);
    printf("[INFO] %dx%d Matrix Partitioning\n\n", horz_partitions, vert_partitions);
    A = coo2dcoo(B, horz_partitions, vert_partitions);
    freeCOOMatrix(B);


    // Initialize help data - Padding needed
    uint32_t ncols_pad = A->vert_partitions * A->tile_width;
    uint32_t tile_width_pad = A->tile_width;
    uint32_t nrows_pad = A->vert_partitions * A->tile_width;
    if (ncols_pad % (8 / byte_dt) != 0)
        ncols_pad = ncols_pad + ((8 / byte_dt) - (ncols_pad % (8 / byte_dt)));
    if (tile_width_pad % (8 / byte_dt) != 0)
        tile_width_pad = tile_width_pad + ((8 / byte_dt) - (tile_width_pad % (8 / byte_dt)));
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
    uint64_t max_rows_per_dpu = 0;
    uint64_t max_nnz_per_dpu = 0;

    // Timer for measurements
    Timer timer;

    uint64_t total_nnzs = 0;
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Find padding for rows and non-zero elements needed for CPU-DPU transfers
        uint32_t tile_row_indx = i / A->vert_partitions; 
        uint32_t tile_col_indx = i % A->vert_partitions; 
        uint32_t rows_per_dpu = A->tile_height; 
        uint32_t prev_rows_dpu = tile_row_indx * A->tile_height;
        if (rows_per_dpu > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu;

        unsigned int nnz=0, nnz_pad;
        nnz = A->nnzs_per_partition[i];
        if (nnz % (8 / byte_dt) != 0)
            nnz_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
        else
            nnz_pad = nnz;

        if (nnz_pad > max_nnz_per_dpu)
            max_nnz_per_dpu = nnz_pad;

        uint32_t prev_nnz_dpu = total_nnzs;
        total_nnzs += nnz;

        // Keep information per DPU
        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].prev_nnz_dpu = prev_nnz_dpu;
        dpu_info[i].nnz = nnz;
        dpu_info[i].nnz_pad = nnz_pad;

        // Find input arguments per DPU
        input_args[i].tcols = tile_width_pad; 

        // Distribute NNZ among tasklets within DPU
        for(unsigned int tasklet_id=0; tasklet_id < NR_TASKLETS; tasklet_id++) {
            uint32_t nnz_chunks = nnz / NR_TASKLETS;
            uint32_t rest_nnzs = nnz % NR_TASKLETS;
            uint32_t nnz_per_tasklet = nnz_chunks;
            uint32_t prev_nnz;

            if (tasklet_id < rest_nnzs)
                nnz_per_tasklet++;
            if (rest_nnzs > 0) {
                if (tasklet_id >= rest_nnzs)
                    prev_nnz = rest_nnzs * (nnz_chunks + 1) + (tasklet_id - rest_nnzs) * nnz_chunks;
                else
                    prev_nnz = tasklet_id * (nnz_chunks + 1);
            } else {
                prev_nnz = tasklet_id * nnz_chunks;
            }
            input_args[i].start_nnz[tasklet_id] = prev_nnz; 
            input_args[i].nnz_per_tasklet[tasklet_id] = nnz_per_tasklet; 
        }
    }

    // Initializations for parallel transfers with padding needed
    if (max_rows_per_dpu % (8 / byte_dt)  != 0) 
        max_rows_per_dpu += ((8 / byte_dt) - (max_rows_per_dpu % (8 / byte_dt)));
    if (max_nnz_per_dpu % (8 / byte_dt) != 0)
        max_nnz_per_dpu += ((8 / byte_dt) - (max_nnz_per_dpu % (8 / byte_dt)));

    // Re-allocations for padding needed
    A->nnzs = (struct elem_t *) realloc(A->nnzs, (dpu_info[nr_of_dpus-1].prev_nnz_dpu + max_nnz_per_dpu) * sizeof(struct elem_t));
    y = (val_dt *) calloc((uint64_t) ((uint64_t) nr_of_dpus) * ((uint64_t) max_rows_per_dpu), sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_nnz_per_dpu) * sizeof(struct elem_t)) + (tile_width_pad * sizeof(val_dt)) + (max_rows_per_dpu * sizeof(val_dt));
    assert(total_bytes <= DPU_CAPACITY && "Bytes needed exceeded MRAM size");


    // Copy input arguments to DPUs
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        input_args[i].max_rows_per_dpu = max_rows_per_dpu; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));


    // Copy input matrix to DPUs
    startTimer(&timer, 0);

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->nnzs + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + tile_width_pad * sizeof(val_dt), max_nnz_per_dpu * sizeof(struct elem_t), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);



    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i % A->vert_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + tile_vert_indx * A->tile_width));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), tile_width_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
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
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);


    // Merge partial results to the host CPU
    startTimer(&timer, 4);
    uint32_t r, c, t;
#pragma omp parallel for num_threads(p.nthreads) shared(A, y, max_rows_per_dpu) private(r,c,t) collapse(2) 
    for (r = 0; r < A->horz_partitions; r++) {
        for (t = 0; t < A->tile_height; t++) {
            for (c = 1; c < A->vert_partitions; c++) {
                y[r * A->vert_partitions * max_rows_per_dpu + t] += y[r * A->vert_partitions * max_rows_per_dpu + c * max_rows_per_dpu + t];
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
            if((r * A->tile_height + t < A->nrows) && y_host[i] != y[r * A->vert_partitions * max_rows_per_dpu + t]) {
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
    freeDCOOMatrix(A);
    free(x);
    free(y);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
