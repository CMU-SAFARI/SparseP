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
static struct RBDBCSRMatrix* A;
static struct RBDCSRMatrix* B;
static struct COOMatrix* C;
static val_dt* x;
static val_dt* y;
static val_dt* z;
static struct partition_info_t *part_info;


/**
 * @brief Specific information for each DPU
 */
struct dpu_info_t {
    uint32_t block_rows_per_dpu;
    uint32_t prev_block_rows_dpu;
    uint32_t cols_per_dpu;
    uint32_t block_start;
    uint32_t blocks;
    uint32_t blocks_pad;
    uint32_t prev_blocks_dpu;
    uint32_t ptr_offset;
    uint32_t merge;
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
static void spmv_host(val_dt* y, struct RBDBCSRMatrix *A, val_dt* x) {

    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < A->vert_partitions; c++) {

        uint32_t ptr_offset = c * (A->num_block_rows + 1);
        uint32_t col_offset = c * A->tile_width;
        for(uint64_t n=0; n < A->num_block_rows; n++) {
            for(uint64_t i=A->browptr[ptr_offset + n]; i<A->browptr[ptr_offset + n+1]; i++){
                uint64_t j = A->bcolind[total_blocks + i];
                for(uint64_t blr=0; blr < A->row_block_size; blr++){
                    val_dt acc = 0;
                    for(uint64_t blc=0; blc < A->col_block_size; blc++) {
                        acc += A->bval[(total_blocks + i) * A->col_block_size * A->row_block_size + blr * A->col_block_size + blc] * x[col_offset + j * A->col_block_size + blc];

                    }
                    y[n * A->row_block_size + blr] += acc;
                }
            }
        }

        total_blocks += A->blocks_per_vert_partition[c];
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
    C = readCOOMatrix(p.fileName);
    sortCOOMatrix(C);
    uint32_t horz_partitions = 0;
    uint32_t vert_partitions = p.vert_partitions; 
    find_partitions(nr_of_dpus, &horz_partitions, p.vert_partitions);
    printf("[INFO] %dx%d Matrix Partitioning\n\n", horz_partitions, vert_partitions);
    B = coo2rbdcsr(C, horz_partitions, vert_partitions);
    freeCOOMatrix(C);
    A = rbdcsr2rbdbcsr(B, p.row_blsize, p.col_blsize);
    countNNZperBlockRBDBCSRMatrix(A);
    freeRBDCSRMatrix(B);

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
    uint32_t ncols_pad = A->ncols + A->tile_width + A->col_block_size;
    uint32_t tile_width_pad = A->num_block_cols * A->col_block_size;
    uint32_t nrows_pad = A->nrows + A->row_block_size; 
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

    // Allocate output vector
    z = (val_dt *) calloc(nrows_pad, sizeof(val_dt)); 

    // Initialize input vector with arbitrary data
    init_vector(x, ncols_pad);

    // Load-balance nnzs (at block-row granularity) across DPUs of the same vertical partition
    partition_by_nnz(A, part_info);

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t)); 
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    // Max limits for parallel transfers
    uint64_t max_block_rows_per_dpu = 0;
    uint64_t max_blocks_per_dpu = 0;


    // Timer for measurements
    Timer timer;

    i = 0;
    uint32_t acc_blocks = 0;
    uint32_t total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Find padding for block rows and non-zero elements needed for CPU-DPU transfers
        uint32_t tile_horz_indx = i % A->horz_partitions; 
        uint32_t tile_vert_indx = i / A->horz_partitions; 
        uint32_t block_rows_per_dpu = part_info->brow_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx + 1] - part_info->brow_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx];
        uint32_t block_rows_per_dpu_pad = part_info->brow_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx + 1] - part_info->brow_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx] + 1;
        uint32_t prev_block_rows_dpu = part_info->brow_split[tile_vert_indx * (A->horz_partitions + 1) + tile_horz_indx];


        if (block_rows_per_dpu_pad > max_block_rows_per_dpu)
            max_block_rows_per_dpu = block_rows_per_dpu_pad;

        unsigned int blocks, blocks_pad;
        blocks = A->browptr[tile_vert_indx * (A->num_block_rows + 1) + prev_block_rows_dpu + block_rows_per_dpu] - A->browptr[tile_vert_indx * (A->num_block_rows + 1) + prev_block_rows_dpu];
        assert(blocks == part_info->blocks_dpu[i]);
        if (blocks % 2 != 0) // bcolind 
            blocks_pad = blocks + 1;
        else
            blocks_pad = blocks;

        if (blocks_pad > max_blocks_per_dpu)
            max_blocks_per_dpu = blocks_pad;

        // Keep information per DPU
        dpu_info[i].block_rows_per_dpu = block_rows_per_dpu;
        dpu_info[i].prev_block_rows_dpu = prev_block_rows_dpu;
        dpu_info[i].cols_per_dpu = A->tile_width;
        dpu_info[i].blocks = blocks;
        dpu_info[i].blocks_pad = blocks_pad;
        dpu_info[i].prev_blocks_dpu = total_blocks;
        dpu_info[i].ptr_offset = tile_vert_indx * (A->num_block_rows + 1) + prev_block_rows_dpu;

        // Find input arguments per DPU
        input_args[i].block_rows = block_rows_per_dpu;
        input_args[i].tcols = tile_width_pad; 
        input_args[i].row_block_size = A->row_block_size; 
        input_args[i].col_block_size = A->col_block_size; 

#if BLNC_TSKLT_BLOCK
        // Load-balance blocks across tasklets 
        partition_tsklt_by_block(A, part_info, i, NR_TASKLETS, nr_of_dpus, acc_blocks, prev_block_rows_dpu, block_rows_per_dpu, tile_vert_indx);
#else
        // Load-balance nnzs across tasklets 
        partition_tsklt_by_nnz(A, part_info, i, NR_TASKLETS, nr_of_dpus, acc_blocks, prev_block_rows_dpu, block_rows_per_dpu, tile_vert_indx);
#endif

        uint32_t t;
        for (t = 0; t < NR_TASKLETS; t++) {
            // Find input arguments per tasklet
            input_args[i].start_block_row[t] = part_info->brow_split_tasklet[i * (NR_TASKLETS+2) + t]; 
            input_args[i].end_block_row[t] = part_info->brow_split_tasklet[i * (NR_TASKLETS+2) + (t+1)]; 
        }

        if (tile_horz_indx == (A->horz_partitions - 1))
            acc_blocks += A->blocks_per_vert_partition[tile_vert_indx];
        total_blocks += part_info->blocks_dpu[i]; 
    }

#if FG_TRANS
    // Find max number of block rows (subset of elements of the output vector) among DPUs of each rank
    DPU_RANK_FOREACH(dpu_set, rank, each_rank){
        uint32_t max_block_rows_cur_rank = 0;
        uint32_t nr_dpus_in_rank;
        DPU_ASSERT(dpu_get_nr_dpus(rank, &nr_dpus_in_rank));
        uint32_t start_dpu = part_info->accum_dpus_ranks[each_rank];
        for (uint32_t k = 0; k < nr_dpus_in_rank; k++) {
            if (start_dpu + k >= nr_of_dpus)
                break;
            if (dpu_info[start_dpu + k].block_rows_per_dpu > max_block_rows_cur_rank)
                max_block_rows_cur_rank =  dpu_info[start_dpu + k].block_rows_per_dpu;

        }

        if (max_block_rows_cur_rank % 2 != 0) 
            max_block_rows_cur_rank++;
        part_info->max_block_rows_per_rank[each_rank] = (uint32_t) max_block_rows_cur_rank;
    }
#endif


    // Initializations for parallel transfers with padding needed
    if (max_block_rows_per_dpu % 2 != 0)
        max_block_rows_per_dpu++;
    if (max_blocks_per_dpu % 2 != 0)
        max_blocks_per_dpu++;

    // Re-allocations for padding needed
    A->browptr = (uint32_t *) realloc(A->browptr, (max_block_rows_per_dpu * nr_of_dpus * sizeof(uint32_t)));
    A->bcolind = (uint32_t *) realloc(A->bcolind, (max_blocks_per_dpu * nr_of_dpus * sizeof(uint32_t)));
    A->bval = (val_dt *) realloc(A->bval, (max_blocks_per_dpu * A->row_block_size * A->col_block_size * nr_of_dpus * sizeof(val_dt)));
    y = (val_dt *) calloc((uint64_t) ((uint64_t) nr_of_dpus * (uint64_t) max_block_rows_per_dpu * A->row_block_size), sizeof(val_dt)); 

    // Count total number of bytes to be transfered in MRAM of DPU
    unsigned long int total_bytes;
    total_bytes = ((max_block_rows_per_dpu) * sizeof(uint32_t)) + (max_blocks_per_dpu * sizeof(uint32_t)) + (max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt)) + (tile_width_pad * sizeof(val_dt)) + (max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt));
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

    // Copy Browptr 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->browptr + dpu_info[i].ptr_offset));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, (max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + tile_width_pad * sizeof(val_dt)), max_block_rows_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));

    // Copy Bcolind
    i = 0;
    total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bcolind + total_blocks));
        total_blocks += part_info->blocks_dpu[i];
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + tile_width_pad * sizeof(val_dt) + max_block_rows_per_dpu * sizeof(uint32_t), max_blocks_per_dpu * sizeof(uint32_t), DPU_XFER_DEFAULT));

    // Copy Bvalues
    i = 0;
    total_blocks = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, A->bval + ((uint64_t) total_blocks * A->row_block_size * A->col_block_size)));
        total_blocks += part_info->blocks_dpu[i];
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt) + tile_width_pad * sizeof(val_dt) + max_block_rows_per_dpu * sizeof(uint32_t) + max_blocks_per_dpu * sizeof(uint32_t), max_blocks_per_dpu * A->row_block_size * A->col_block_size * sizeof(val_dt), DPU_XFER_DEFAULT));
    stopTimer(&timer, 0);


    // Copy input vector  to DPUs
    startTimer(&timer, 1);
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t tile_vert_indx = i / A->horz_partitions; 
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
#if CG_TRANS
    // Coarse-grained data transfers in the output vector
    i = 0;
    uint32_t block_rows_footprint = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_block_rows_per_dpu * A->row_block_size)));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_block_rows_per_dpu * A->row_block_size * sizeof(val_dt), DPU_XFER_DEFAULT));
#endif

#if FG_TRANS
    // Fine-grained data transfers in the output vector at rank granularity
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, y + (i * max_block_rows_per_dpu * A->row_block_size)));
    }
    i = 0;
    //struct dpu_set_t rank;
    DPU_RANK_FOREACH(dpu_set, rank) {
        DPU_ASSERT(dpu_push_xfer(rank, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, part_info->max_block_rows_per_rank[i] * A->row_block_size * sizeof(val_dt), DPU_XFER_ASYNC));
        i++;
    }
    DPU_ASSERT(dpu_sync(dpu_set));
#endif
    stopTimer(&timer, 3);


    // Merge partial results to the host CPU
    startTimer(&timer, 4);
    uint32_t r, c, t, b;
    for (c = 0; c < A->vert_partitions; c++) {
        for (r = 0; r < A->horz_partitions; r++) {
#pragma omp parallel for num_threads(p.nthreads) shared(A, z, y, max_block_rows_per_dpu, r, c) private(t, b)
            for (t = 0; t < part_info->brow_split[c * (A->horz_partitions + 1) + r+1] - part_info->brow_split[c * (A->horz_partitions + 1) + r]; t++) {
                for (b = 0; b < A->row_block_size; b++) {
                    z[(part_info->brow_split[c * (A->horz_partitions + 1) + r] + t) * A->row_block_size + b] += y[(c * A->horz_partitions + r) * max_block_rows_per_dpu * A->row_block_size + t * A->row_block_size + b];
                }
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
    freeRBDBCSRMatrix(A);
    free(x);
    free(y);
    free(z);
    partition_free(part_info);
    DPU_ASSERT(dpu_free(dpu_set));

    return 0;
}
