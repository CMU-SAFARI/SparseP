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
#include <math.h>

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
static struct BDCOOMatrix* A;
static struct COOMatrix* B;
static val_dt* x;
static val_dt* y;
static val_dt* z;
static struct partition_info_t *part_info;


/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t rows_per_dpu;
    uint32_t cols_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
    uint32_t prev_nnz_dpu;
    uint32_t nnz;
    uint32_t nnz_pad;
};

struct dpu_info_t *dpu_info;


/**
 * @brief find the dpus_per_row_partition
 * @param factor n to create partitions
 * @param column_partitions to create vert_partitions 
 * @param horz_partitions to return the 2D partitioning
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
 * @brief compute output in the host CPU
 */
static void spmv_host(val_dt* y, struct BDCOOMatrix *A, val_dt* x) {

    uint64_t total_nnzs = 0;
    for (uint32_t c = 0; c < A->vert_partitions; c++) {
        uint32_t col_offset = A->vert_tile_widths[c];
        for(uint32_t n = 0; n < A->nnzs_per_vert_partition[c]; n++) {
            uint32_t rowIndx = A->nnzs[total_nnzs].rowind;    
            uint32_t colIndx = A->nnzs[total_nnzs].colind;    
            val_dt value = A->nnzs[total_nnzs++].val;    
            y[rowIndx] += (value * x[col_offset + colIndx]); 
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
    uint32_t nr_of_ranks;

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &nr_of_ranks));
    printf("[INFO] Allocated %d DPU(s)\n", nr_of_dpus);
    printf("[INFO] Allocated %d Rank(s)\n", nr_of_ranks);
    printf("[INFO] Allocated %d TASKLET(s) per DPU\n", NR_TASKLETS);

    unsigned int i;

    // Initialize input data 
    B = readCOOMatrix(p.fileName);
    sortCOOMatrix(B);
    uint32_t horz_partitions = 0;
    uint32_t vert_partitions = p.vert_partitions; 
    find_partitions(nr_of_dpus, &horz_partitions, p.vert_partitions);
    printf("[INFO] %dx%d Matrix Partitioning\n\n", horz_partitions, vert_partitions);
    A = coo2bdcoo(B, horz_partitions, vert_partitions);
    freeCOOMatrix(B);

    // Initialize partition data
    part_info = partition_init(A, nr_of_dpus, p.max_nranks, NR_TASKLETS);

#if FG_TRANS
    struct dpu_set_t rank;
    uint32_t each_rank;
    DPU_RANK_FOREACH(dpu_set, rank, each_rank){
        uint32_t nr_dpus_in_rank;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_in_rank));
        part_info->active_dpus_per_rank[each_rank+1] = nr_dpus_in_rank;
    }

    int sum = 0;
    for(int i=0; i < p.max_nranks+1; i++) {
        part_info->accum_dpus_ranks[i] = part_info->active_dpus_per_rank[i] + sum;
        sum += part_info->active_dpus_per_rank[i];
    }
#endif


    // Initialize help data - Padding needed
    uint32_t ncols_pad = A->vert_tile_widths[A->vert_partitions-1] + A->max_tile_width;
    uint32_t tile_width_pad = A->max_tile_width;
    uint32_t nrows_pad = A->nrows;
    if (ncols_pad % (8 / byte_dt) != 0)
        ncols_pad = ncols_pad + ((8 / byte_dt) - (ncols_pad % (8 / byte_dt)));
    if (tile_width_pad % (8 / byte_dt) != 0)
        tile_width_pad = tile_width_pad + ((8 / byte_dt) - (tile_width_pad % (8 / byte_dt)));
    if (nrows_pad % (8 / byte_dt) != 0)
        nrows_pad = nrows_pad + ((8 / byte_dt) - (nrows_pad % (8 / byte_dt)));

    // Allocate input vector
    x = (val_dt *) malloc(ncols_pad * sizeof(val_dt)); 

    // Allocate output vector
    z = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 

    // Initialize input vector with arbitrary data
    init_vector(x, ncols_pad);

    // Load-balance nnzs among DPUs of the same vertical partition
    partition_by_nnz(A, part_info);

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
        uint32_t tile_horz_indx = i % A->horz_partitions; 
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        uint32_t rows_per_dpu = part_info->row_split[tile_vert_indx * (2 * A->horz_partitions) + 2 * tile_horz_indx + 1] - part_info->row_split[tile_vert_indx * (2 * A->horz_partitions) + 2 * tile_horz_indx];
        uint32_t prev_rows_dpu = part_info->row_split[tile_vert_indx * (2 * A->horz_partitions) + 2 * tile_horz_indx];

        if (rows_per_dpu > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu;

        uint32_t rows_per_dpu_pad = rows_per_dpu;
        if (rows_per_dpu_pad % (8 / byte_dt)  != 0) 
            rows_per_dpu_pad += ((8 / byte_dt) - (rows_per_dpu_pad % (8 / byte_dt)));

        unsigned int nnz=0, nnz_pad;
        nnz = part_info->nnz_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx + 1] - part_info->nnz_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx];

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
        dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;
        dpu_info[i].cols_per_dpu = A->vert_tile_widths[tile_vert_indx+1] - A->vert_tile_widths[tile_vert_indx];
        dpu_info[i].prev_nnz_dpu = prev_nnz_dpu;
        dpu_info[i].nnz = nnz;
        dpu_info[i].nnz_pad = nnz_pad;

        // Find input arguments per DPU
        input_args[i].nrows = rows_per_dpu;
        input_args[i].tcols = tile_width_pad; 
        input_args[i].tstart_row = prev_rows_dpu; 

        // Load-balance nnzs across tasklets of a DPU
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
            // Find input arguments per tasklet
            input_args[i].start_nnz[tasklet_id] = prev_nnz; 
            input_args[i].nnz_per_tasklet[tasklet_id] = nnz_per_tasklet; 
        }
    }

#if FG_TRANS
    // Find max number of rows and columns (subset of elements of the output vector) among DPUs of each rank
    DPU_RANK_FOREACH(dpu_set, rank, each_rank){
        uint32_t max_rows_cur_rank = 0;
        uint32_t max_cols_cur_rank = 0;
        uint32_t nr_dpus_in_rank;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_in_rank));
        uint32_t start_dpu = part_info->accum_dpus_ranks[each_rank];
        for (uint32_t k = 0; k < nr_dpus_in_rank; k++) {
            if (start_dpu + k >= nr_of_dpus)
                break;
            if (dpu_info[start_dpu + k].rows_per_dpu > max_rows_cur_rank)
                max_rows_cur_rank =  dpu_info[start_dpu + k].rows_per_dpu;
            if (dpu_info[start_dpu + k].cols_per_dpu > max_cols_cur_rank)
                max_cols_cur_rank =  dpu_info[start_dpu + k].cols_per_dpu;

        }

        if (max_rows_cur_rank % (8 / byte_dt)  != 0) 
            max_rows_cur_rank += ((8 / byte_dt) - (max_rows_cur_rank % (8 / byte_dt)));
        if (max_cols_cur_rank % (8 / byte_dt)  != 0) 
            max_cols_cur_rank += ((8 / byte_dt) - (max_cols_cur_rank % (8 / byte_dt)));
        part_info->max_rows_per_rank[each_rank] = (uint32_t) max_rows_cur_rank;
        part_info->max_cols_per_rank[each_rank] = (uint32_t) max_cols_cur_rank;
    }
#endif

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

    // Copy Input Array 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->nnzs + dpu_info[i].prev_nnz_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt) + tile_width_pad * sizeof(val_dt), max_nnz_per_dpu * sizeof(struct elem_t), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);


    // Copy input vector  to DPUs
    startTimer(&timer, 1);
#if CG_TRANS
    // Coarse-grained data transfers in the input vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + A->vert_tile_widths[tile_vert_indx]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), tile_width_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
#endif

#if FG_TRANS
#if YFG_TRANS
    // Coarse-grained data transfers in the input vector at rank granularity
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + A->vert_tile_widths[tile_vert_indx]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), tile_width_pad * sizeof(val_dt), DPU_XFER_DEFAULT));
#else
    // Fine-grained data transfers in the input vector at rank granularity
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        DPU_ASSERT(dpu_prepare_xfer(dpu, x + A->vert_tile_widths[tile_vert_indx]));
    }
    i = 0;
    //struct dpu_set_t rank;
    DPU_RANK_FOREACH(dpu_set, rank) {
        DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * sizeof(val_dt), part_info->max_cols_per_rank[i] * sizeof(val_dt), DPU_XFER_ASYNC));
        i++;
    }
    DPU_ASSERT(dpu_sync(dpu_set));
#endif
#endif
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
#if CG_TRANS
    // Coarse-grained data transfers in the output vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_rows_per_dpu)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * sizeof(val_dt), DPU_XFER_DEFAULT));
#endif

#if FG_TRANS
    // Fine-grained data transfers in the output vector at rank granularity
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + i * max_rows_per_dpu));
    }
    i = 0;
    DPU_RANK_FOREACH(dpu_set, rank) {
        DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, part_info->max_rows_per_rank[i] * sizeof(val_dt), DPU_XFER_ASYNC));
        i++;
    }
    DPU_ASSERT(dpu_sync(dpu_set));
#endif
    stopTimer(&timer, 3);



    // Merge partial results to the host CPU
    startTimer(&timer, 4);
    uint32_t r, c, t;
    for (c = 0; c < A->vert_partitions; c++) {
        for (r = 0; r < A->horz_partitions; r++) {
#pragma omp parallel for num_threads(p.nthreads) shared(A, z, y, max_rows_per_dpu, c, r) private(t)
            for (t = 0; t < part_info->row_split[c * (2 * A->horz_partitions) + 2 * r+1] - part_info->row_split[c * (2 * A->horz_partitions) + 2 * r]; t++) {
                z[part_info->row_split[c * (2 * A->horz_partitions) + 2 * r] + t] += y[(c * A->horz_partitions + r) * max_rows_per_dpu + t];
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
    startTimer(&timer, 4);
    val_dt *y_host = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 
    spmv_host(y_host, A, x); 

    bool status = true;
    i = 0;
    for (i = 0; i < A->nrows; i++) {
        if(y_host[i] != z[i]) {
            status = false;
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
    freeBDCOOMatrix(A);
    free(x);
    free(y);
    free(z);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
