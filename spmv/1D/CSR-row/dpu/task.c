/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"
#include "../support/utils.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
uint32_t nnz_offset;

//#define SEQREAD_CACHE_SIZE 128

/**
 * @brief main function executed by each tasklet
 */
int main() {
    uint32_t tasklet_id = me();

    if (tasklet_id == 0){ 
        mem_reset(); // Reset the heap
    }

    // Barrier
    barrier_wait(&my_barrier);

    // Load parameters
    uint32_t nrows = DPU_INPUT_ARGUMENTS.nrows;
    uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;
    uint32_t max_nnz_ind = DPU_INPUT_ARGUMENTS.max_nnz_ind;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    unsigned int start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];
    unsigned int rows_per_tasklet = DPU_INPUT_ARGUMENTS.rows_per_tasklet[tasklet_id];


    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows * sizeof(val_dt)));
    uint32_t mram_base_addr_rowptr = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_colind = (uint32_t) (mram_base_addr_rowptr + (max_rows * sizeof(uint32_t)));
    uint32_t mram_base_addr_values = (uint32_t) (mram_base_addr_colind + (max_nnz_ind * sizeof(uint32_t)));

    // If there is no work, return
    if (rows_per_tasklet == 0) {
        goto EXIT;
    }

    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(8);

    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(8);

    // Initialize sequential reader for rowptr
    uint64_t temp;
    mram_read((__mram_ptr void const *) (mram_base_addr_rowptr), (void *) (&temp), 8);
    nnz_offset = (uint32_t) temp; 
    mram_base_addr_rowptr += (start_row * sizeof(uint32_t));
    seqreader_buffer_t cache_rowptr = seqread_alloc();
    seqreader_t sr_rowptr;

    // Find first row per tasklet
    uint32_t *current_row = seqread_init(cache_rowptr, (__mram_ptr void *) mram_base_addr_rowptr, &sr_rowptr);
    uint32_t prev_row = *current_row;

    // Find MRAM addreses for colind, values, y vector per tasklet
    mram_base_addr_colind += ((prev_row - nnz_offset) * sizeof(uint32_t));
    mram_base_addr_values += ((prev_row - nnz_offset) * sizeof(val_dt));
    mram_base_addr_y += (start_row * sizeof(val_dt)); 

    // Initialize sequential readers for colind and values
    seqreader_buffer_t cache_colind = seqread_alloc();
    seqreader_buffer_t cache_val = seqread_alloc();
    seqreader_t sr_colind;
    seqreader_t sr_val;
    uint32_t *current_colind = seqread_init(cache_colind, (__mram_ptr void *) mram_base_addr_colind, &sr_colind);
    val_dt *current_val = seqread_init(cache_val, (__mram_ptr void *) mram_base_addr_values, &sr_val);

    // Initialize help variables
    uint32_t i, j;
    uint32_t write_indx = 0;
    uint32_t write_bound = 8 / byte_dt;
    val_dt acc;

    // Check
    if (start_row + rows_per_tasklet > nrows)
        rows_per_tasklet = nrows - start_row;

    // SpMV 
    // Iterate over rows
    for (i=start_row; i < start_row + rows_per_tasklet; i++) {
        current_row = seqread_get(current_row, sizeof(*current_row), &sr_rowptr);
        acc = 0;

        // Iterate over non-zero elements for each row
        // For each non-zero element: 1. get input vector value, 2. multiply and add
        for (j=0; j < *current_row - prev_row; j++) {
#if INT8
            if (((*current_colind) & 7) == 0) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[0];
            } else if (((*current_colind) & 7) == 1) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 1) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[1];
            } else if (((*current_colind) & 7) == 2) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 2) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[2];
            } else if (((*current_colind) & 7) == 3) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 3) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[3];
            } else if (((*current_colind) & 7) == 4) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 4) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[4];
            } else if (((*current_colind) & 7) == 5) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 5) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[5];
            } else if (((*current_colind) & 7) == 6) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 6) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[6];
            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 7) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[7];
            }
#elif INT16
            if (((*current_colind) & 3) == 0) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[0];
            } else if (((*current_colind) & 3) == 1) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 1) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[1];
            } else if (((*current_colind) & 3) == 2) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 2) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[2];
            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 3) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[3];
            } 
#elif INT32
            if (((*current_colind) & 1) == 0) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[0];
            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 1) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[1];
            } 
#elif INT64
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*current_val) * cache_x[0];
#elif FP32
            if (((*current_colind) & 1) == 0) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[0];
            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 1) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[1];
            } 
#elif FP64
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
            acc += (*current_val) * cache_x[0];
#else
            if (((*current_colind) & 1) == 0) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + (*current_colind) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[0];
            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_colind) - 1) * sizeof(val_dt)), cache_x, 8);
                acc += (*current_val) * cache_x[1];
            } 
#endif

            // Read next non-zero element
            current_colind = seqread_get(current_colind, sizeof(*current_colind), &sr_colind);
            current_val = seqread_get(current_val, sizeof(*current_val), &sr_val);
        }

        // Store output
        // Accumulate the output values in WRAM until 8-byte alignment (write_bound) is reached
        // When 8-byte alignement is satisfied, write the output values to MRAM
        if (write_indx != write_bound - 1) {
            cache_y[write_indx] = acc;
            write_indx++;
        } else {
            cache_y[write_indx] = acc;
            write_indx = 0;
            mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
            mram_base_addr_y += 8;
        }
        prev_row = *current_row;
    }

    // Store output for the last output vector element to MRAM, if needed
    if (write_indx != 0) {
        for (i=write_indx; i < write_bound; i++) 
            cache_y[i] = 0;
        mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
    }


EXIT: 

    return 0;
}
