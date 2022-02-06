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
static struct COOMatrix* A;

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
 * @brief compute output in the host CPU
 */
static void spmv_host(val_dt* y, struct COOMatrix *A, val_dt* x) {

    for(unsigned int n = 0; n < A->nnz; n++) {
        y[A->nnzs[n].rowind] += x[A->nnzs[n].colind] * A->nnzs[n].val;
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
    A = readCOOMatrix(p.fileName);
    sortCOOMatrix(A);

    // Allocate input vector
    x = (val_dt *) malloc(A->ncols * sizeof(val_dt)); 
    // Initialize input vector with arbitrary data
    init_vector(x, A->ncols);

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_rows_per_dpu = 0;
    uint64_t max_nnz_per_dpu = 0;


    // Timer for measurements
    Timer timer;

    uint32_t prev_row = 0;
    uint32_t prev_nnz = 0;
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Distribute NNZ to DPUs
        uint32_t nnz_chunks = A->nnz / nr_of_dpus;
        uint32_t rest_nnzs = A->nnz % nr_of_dpus;
        uint32_t nnz_per_dpu = nnz_chunks;
        uint32_t prev_nnz_dpu;

        if (i < rest_nnzs)
            nnz_per_dpu++;
        if (rest_nnzs > 0) {
            if (i >= rest_nnzs)
                prev_nnz_dpu = rest_nnzs * (nnz_chunks + 1) + (i - rest_nnzs) * nnz_chunks;
            else
                prev_nnz_dpu = i * (nnz_chunks + 1);
        } else {
            prev_nnz_dpu = i * nnz_chunks;
        }

        uint32_t prev_rows_dpu = prev_row;
        uint32_t r = prev_row;
        uint32_t t = prev_nnz;
        uint32_t cur_nnz = 0;
        while (cur_nnz < nnz_per_dpu) {
            cur_nnz += A->rows[r] - t;
            t = 0;
            r++;
        }
        uint32_t rows_per_dpu = r - prev_row; 
        if (prev_nnz == 0)
            dpu_info[i].merge = 0;
        else
            dpu_info[i].merge = 1;

        prev_row = r-1;
        prev_nnz = A->rows[r-1] - (cur_nnz - nnz_per_dpu);
        if (cur_nnz == nnz_per_dpu) {
            prev_row = r;
            prev_nnz = 0;
        }

        // Pad data to be transfered for rows and nnzs
        if (rows_per_dpu > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu;

        unsigned int nnz, nnz_pad;
        nnz = nnz_per_dpu;
        if (nnz % (8 / byte_dt) != 0)
            nnz_pad = nnz + ((8 / byte_dt) - (nnz % (8 / byte_dt)));
        else
            nnz_pad = nnz;

        if (nnz_pad > max_nnz_per_dpu)
            max_nnz_per_dpu = nnz_pad;

        // Keep information per DPU
        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].prev_nnz_dpu = prev_nnz_dpu;
        dpu_info[i].nnz = nnz;
        dpu_info[i].nnz_pad = nnz_pad;

        // Find input arguments per DPU
        input_args[i].nrows = rows_per_dpu;
        input_args[i].tcols = A->ncols; 
        input_args[i].tstart_row = dpu_info[i].prev_rows_dpu;

        // Distribute NNZ across tasklets of a DPU
        // Load-balance nnz across tasklets 
        for(unsigned int tasklet_id=0; tasklet_id < NR_TASKLETS; tasklet_id++) {
            uint32_t nnz_chunks_tskl = nnz / NR_TASKLETS;
            uint32_t rest_nnzs_tskl = nnz % NR_TASKLETS;
            uint32_t nnz_per_tasklet = nnz_chunks_tskl;
            uint32_t prev_nnz_tskl;

            if (tasklet_id < rest_nnzs_tskl)
                nnz_per_tasklet++;
            if (rest_nnzs_tskl > 0) {
                if (tasklet_id >= rest_nnzs_tskl)
                    prev_nnz_tskl = rest_nnzs_tskl * (nnz_chunks_tskl + 1) + (tasklet_id - rest_nnzs_tskl) * nnz_chunks_tskl;
                else
                    prev_nnz_tskl = tasklet_id * (nnz_chunks_tskl + 1);
            } else {
                prev_nnz_tskl = tasklet_id * nnz_chunks_tskl;
            }
            input_args[i].start_nnz[tasklet_id] = prev_nnz_tskl; 
            input_args[i].nnz_per_tasklet[tasklet_id] = nnz_per_tasklet; 

        }
    }

    // Initializations for parallel transfers with padding needed
    if (max_rows_per_dpu % (8 / byte_dt)  != 0) 
        max_rows_per_dpu += ((8 / byte_dt) - (max_rows_per_dpu % (8 / byte_dt)));
    if (max_nnz_per_dpu % (8 / byte_dt) != 0)
        max_nnz_per_dpu += ((8 / byte_dt) - (max_nnz_per_dpu % (8 / byte_dt)));

    // Re-allocations
    A->nnzs = (struct elem_t *) realloc(A->nnzs, (max_nnz_per_dpu) * nr_of_dpus * sizeof(struct elem_t));
    y = (val_dt *) calloc((uint64_t) ((uint64_t) nr_of_dpus) * ((uint64_t) max_rows_per_dpu), sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_nnz_per_dpu) * sizeof(struct elem_t)) + (A->ncols * sizeof(val_dt)) + (max_rows_per_dpu * sizeof(val_dt));
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
    // Copy input array
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->nnzs + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + A->ncols * sizeof(val_dt), max_nnz_per_dpu * sizeof(struct elem_t), DPU_XFER_DEFAULT));
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
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 3);


    // Merge partial results to the host CPU
    startTimer(&timer, 4);
    unsigned int t, n = 0;
    while (n < nr_of_dpus - 1) {
        if (dpu_info[n].prev_rows_dpu + dpu_info[n].rows_per_dpu - 1 == dpu_info[n+1].prev_rows_dpu) {
            t = n;
            // Merge multiple y[i] elements computed in different DPUs
            while (dpu_info[n].prev_rows_dpu + dpu_info[n].rows_per_dpu - 1 == dpu_info[t+1].prev_rows_dpu) {
                y[n * max_rows_per_dpu + dpu_info[n].rows_per_dpu - 1] += y[(t+1) * max_rows_per_dpu];
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
    val_dt *y_host = (val_dt *) calloc((A->nrows), sizeof(val_dt)); 
    spmv_host(y_host, A, x); 

    bool status = true;
    i = 0;
    unsigned int j;
    for (n = 0; n < nr_of_dpus; n++) {
        for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
            if(j == 0 && dpu_info[n].merge == 1) 
                continue; 
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
    freeCOOMatrix(A);
    free(x);
    free(y);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
