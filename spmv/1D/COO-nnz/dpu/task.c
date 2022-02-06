/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 */

#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <mutex.h>
#include <seqread.h>

#include "../support/common.h"
#include "../support/utils.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

#if LOCKFREE
// LOCK-FREE implementation


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

uint32_t *sync_rptr; // Global cache memory to temporarily store indexes of the partial results created for the output vector elements 
val_dt *sync_values; // Global cache memory to temporarily store values of the partial results created for the output vector elements 


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
    uint32_t max_rows_per_dpu = DPU_INPUT_ARGUMENTS.max_rows_per_dpu;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
    uint32_t start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
    uint32_t nnz_per_tasklet = DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_temp_addr_y;
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows_per_dpu * sizeof(val_dt)));
    uint32_t mram_base_addr_elems = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));


    uint32_t i, j;
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(8);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(8);

    // Use cache_y cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
#if INT8
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
        cache_y[4] = 0;
        cache_y[5] = 0;
        cache_y[6] = 0;
        cache_y[7] = 0;
#elif INT16
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
#elif INT32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif INT64
        cache_y[0] = 0;
#elif FP32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif FP64
        cache_y[0] = 0;
#else
        cache_y[0] = 0;
        cache_y[1] = 0;
#endif

        uint32_t iter = 0;
#if INT8
        iter = (max_rows_per_dpu >> 3);
#elif INT16
        iter = (max_rows_per_dpu >> 2);
#elif INT32
        iter = (max_rows_per_dpu >> 1);
#elif INT64
        iter = max_rows_per_dpu;
#elif FP32
        iter = (max_rows_per_dpu >> 1);
#elif FP64
        iter = max_rows_per_dpu;
#else
        iter = (max_rows_per_dpu >> 1);
#endif
        for(i=0; i < iter; i++) {
            mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
            mram_base_addr_y += 8;
        }

        // Allocate cache memory to temporarily store results for the output vector elements
        sync_rptr = mem_alloc(NR_TASKLETS * 8);
        sync_values = mem_alloc(NR_TASKLETS * 8);
        iter = NR_TASKLETS * (8 / byte_dt);
        for(i=0; i < iter; i++) {
            sync_rptr[i] = 0;
            sync_values[i] = 0;
        }
    }
    barrier_wait(&my_barrier);

    // Revert back
    mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);

    // Initialize sequential reader for nnzs
    mram_base_addr_elems += (start_nnz * sizeof(struct elem_t));
    seqreader_buffer_t cache_elems = seqread_alloc();
    seqreader_t sr_elem;
    struct elem_t *cur_elem = seqread_init(cache_elems, (__mram_ptr void *) mram_base_addr_elems, &sr_elem);
    uint32_t prev_row = cur_elem->rowind;

    // Initialize help variables
    uint32_t diff;
    val_dt acc = 0;
    // row_indx and row_bound are used to find how many elements are critical (i.e., mutliple tasklets confict and produce partial results sharing these elements)
    uint32_t row_bound = 8 / byte_dt; 
    uint32_t row_indx = 0; 
    uint32_t sync_base = tasklet_id * row_bound;


    // Find indexes of critical elements shared among multiple tasklets
    // Note the 8-byte alignement: critical elements are also the ones located in the same 8-byte aligned MRAM location
#if INT8
    diff = prev_row - tstart_row;
    row_indx = (diff & 7);
    sync_rptr[tasklet_id] = diff - (diff & 7);
#elif INT16
    diff = prev_row - tstart_row;
    row_indx = (diff & 3);
    sync_rptr[tasklet_id] = diff - (diff & 3);
#elif INT32
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#elif INT64
    diff = prev_row - tstart_row;
    sync_rptr[tasklet_id] = diff;
#elif FP32
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#elif FP64
    diff = prev_row - tstart_row;
    sync_rptr[tasklet_id] = diff;
#else
    diff = prev_row - tstart_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
#endif


    // SpMV for critical elements
    // Iterate over nnzs
    for(i=0; i<nnz_per_tasklet; i++) {
        // Temporarily store values of the critical elements to sync_values cache
        if(cur_elem->rowind != prev_row) {
            sync_values[sync_base + row_indx] = acc;  
            acc = 0;

            // If the next output vector element is calculated by only one tasklet (not critical element),
            // break this loop and move to the main loop to directly store (final) results to MRAM 
            row_indx += (cur_elem->rowind - prev_row);
            if(row_indx >= row_bound) {
                prev_row = cur_elem->rowind;
                break;
            }
            prev_row = cur_elem->rowind;

        }

        // For each non-zero element: 1. get input vector value, 2. multiply and add
#if INT8
        if ((cur_elem->colind & 7) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 7) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 7) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else if ((cur_elem->colind & 7) == 3) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } else if ((cur_elem->colind & 7) == 4) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 4) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[4];
        } else if ((cur_elem->colind & 7) == 5) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 5) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[5];
        } else if ((cur_elem->colind & 7) == 6) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 6) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[6];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 7) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[7];
        }
#elif INT16
        if ((cur_elem->colind & 3) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 3) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 3) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } 
#elif INT32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif INT64
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#elif FP32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif FP64 
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#else
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#endif

        // Get next non-zero element
        cur_elem = seqread_get(cur_elem, sizeof(struct elem_t), &sr_elem);
    }


    // Store output for the last critical output vector element in sync_values cache
    if (i == nnz_per_tasklet && row_indx < row_bound) {
        sync_values[sync_base + row_indx] = acc;  
        acc = 0;
    }

    // SpMV
    // Iterate over nnzs
    for(i; i<nnz_per_tasklet; i++) {
        // If all nnzs of the same row have been traversed, store the final value for the output vector element in MRAM (8-byte alignment to MRAM accesses)
        // Lock-free MRAM accesses
        if(cur_elem->rowind != prev_row) {
            diff = prev_row - tstart_row;
#if INT8
            if ((diff & 7) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 3) {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 4) {
                diff -= 4; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[4] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 5) {
                diff -= 5; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[5] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 6) {
                diff -= 6; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[6] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 7; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[7] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT16
            if ((diff & 3) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT32
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif INT64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#elif FP32
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#elif FP64 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#else
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }

#endif
            acc = 0;
            prev_row = cur_elem->rowind;
        }

        // For each non-zero element: 1. get input vector value, 2. multiply and add
#if INT8
        if ((cur_elem->colind & 7) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 7) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 7) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else if ((cur_elem->colind & 7) == 3) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } else if ((cur_elem->colind & 7) == 4) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 4) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[4];
        } else if ((cur_elem->colind & 7) == 5) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 5) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[5];
        } else if ((cur_elem->colind & 7) == 6) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 6) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[6];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 7) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[7];
        }
#elif INT16
        if ((cur_elem->colind & 3) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 3) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 3) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } 
#elif INT32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif INT64
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#elif FP32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif FP64 
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#else
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#endif

        // Get next non-zero element
        cur_elem = seqread_get(cur_elem, sizeof(struct elem_t), &sr_elem);
    }


    // Store output for the last output vector element in MRAM
    // Lock-free MRAM accesses
    if (row_indx >= row_bound) {
        diff = prev_row - tstart_row;
#if INT8
        if ((diff & 7) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 1) {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 2) {
            diff -= 2; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[2] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 3) {
            diff -= 3; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[3] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 4) {
            diff -= 4; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[4] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 5) {
            diff -= 5; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[5] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 7) == 6) {
            diff -= 6; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[6] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 7; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[7] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT16
        if ((diff & 3) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 3) == 1) {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else if ((diff & 3) == 2) {
            diff -= 2; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[2] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 3; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[3] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT32
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif INT64
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#elif FP32
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }

#elif FP64 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

#else
        if ((diff & 1) == 0) {
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        } else {
            diff -= 1; 
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[1] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }
#endif

    }


    // Wait for all tasklets to finish
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        // One tasklet merges the partial results that are temporarily stored in sync_values cache
        // Based on their indexes stored in sync_rptr cache one tasklet writes the final output vector elements values to MRAM
        uint32_t t = 0;
        uint32_t iter = 8 / byte_dt;
        // For each tasklet find the partial results that it has created and merge them in MRAM
        for(i = 0; i < NR_TASKLETS; i++) {
#if INT8
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i])); 
#elif INT16
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 1)); 
#elif INT32
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#elif INT64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 3)); 
#elif FP32
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#elif FP64
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 3)); 
#else
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] << 2)); 
#endif
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            for(j=0; j < iter; j++) {
                if(sync_values[t] != 0)
                    cache_y[j] += sync_values[t]; 
                t++;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }
    }


    return 0;
}






#else
// LOCK-Based implementations (i.e., CGLOCK and FGLOCK)



// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
#if CGLOCK
MUTEX_INIT(my_mutex);
#elif FGLOCK
mutex_id_t my_mutex[32];
#endif

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
    uint32_t max_rows_per_dpu = DPU_INPUT_ARGUMENTS.max_rows_per_dpu;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t tstart_row = DPU_INPUT_ARGUMENTS.tstart_row;
    uint32_t start_nnz = DPU_INPUT_ARGUMENTS.start_nnz[tasklet_id];
    uint32_t nnz_per_tasklet = DPU_INPUT_ARGUMENTS.nnz_per_tasklet[tasklet_id];


    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_temp_addr_y;
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_rows_per_dpu * sizeof(val_dt)));
    uint32_t mram_base_addr_elems = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));


    uint32_t i;
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(8);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(8);

    // Use cache_y cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
#if INT8
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
        cache_y[4] = 0;
        cache_y[5] = 0;
        cache_y[6] = 0;
        cache_y[7] = 0;
#elif INT16
        cache_y[0] = 0;
        cache_y[1] = 0;
        cache_y[2] = 0;
        cache_y[3] = 0;
#elif INT32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif INT64
        cache_y[0] = 0;
#elif FP32
        cache_y[0] = 0;
        cache_y[1] = 0;
#elif FP64
        cache_y[0] = 0;
#else
        cache_y[0] = 0;
        cache_y[1] = 0;
#endif

        uint32_t iter = 0;
#if INT8
        iter = (max_rows_per_dpu >> 3);
#elif INT16
        iter = (max_rows_per_dpu >> 2);
#elif INT32
        iter = (max_rows_per_dpu >> 1);
#elif INT64
        iter = max_rows_per_dpu;
#elif FP32
        iter = (max_rows_per_dpu >> 1);
#elif FP64
        iter = max_rows_per_dpu;
#else
        iter = (max_rows_per_dpu >> 1);
#endif
        for(i=0; i < iter; i++) {
            mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
            mram_base_addr_y += 8;
        }
    }
    barrier_wait(&my_barrier);

    // Revert back
    mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);

    // If there is no work, return
    if (nnz_per_tasklet == 0) {
        goto EXIT;
    }

    // Initialize sequential reader for nnzs
    mram_base_addr_elems += (start_nnz * sizeof(struct elem_t));
    seqreader_buffer_t cache_elems = seqread_alloc();
    seqreader_t sr_elem;
    struct elem_t *cur_elem = seqread_init(cache_elems, (__mram_ptr void *) mram_base_addr_elems, &sr_elem);
    uint32_t prev_row = cur_elem->rowind;

    // Initialize help variables
    uint32_t diff;
    val_dt acc = 0;

    // SpMV
    // Iterate over nnzs
    for(i=0; i<nnz_per_tasklet; i++) {
        // If all nnzs of the same row have been traversed, store the final value for the output vector element in MRAM (8-byte alignment to MRAM accesses)
        if(cur_elem->rowind != prev_row) {
            diff = prev_row - tstart_row;
            // Acquire the lock for MRAM writes
#if INT8
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff >> 3;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            if ((diff & 7) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 3) {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 4) {
                diff -= 4; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[4] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 5) {
                diff -= 5; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[5] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 7) == 6) {
                diff -= 6; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[6] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 7; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[7] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT16
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff >> 2;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            if ((diff & 3) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 1) {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else if ((diff & 3) == 2) {
                diff -= 2; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[2] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 3; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[3] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT32
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff >> 1;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT64
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#elif FP32
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff >> 1;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#elif FP64 
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            cache_y[0] += acc;
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#else
#if CGLOCK
            mutex_lock(my_mutex);
#elif FGLOCK
            uint32_t lock_id = diff >> 1;
            lock_id = lock_id & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            if ((diff & 1) == 0) {
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[0] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            } else {
                diff -= 1; 
                mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
                mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
                cache_y[1] += acc;
                mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
            }
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif

#endif
            acc = 0;
            prev_row = cur_elem->rowind;
        }

        // For each non-zero element: 1. get input vector value, 2. multiply and add
#if INT8
        if ((cur_elem->colind & 7) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 7) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 7) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else if ((cur_elem->colind & 7) == 3) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } else if ((cur_elem->colind & 7) == 4) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 4) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[4];
        } else if ((cur_elem->colind & 7) == 5) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 5) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[5];
        } else if ((cur_elem->colind & 7) == 6) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 6) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[6];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 7) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[7];
        }
#elif INT16
        if ((cur_elem->colind & 3) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else if ((cur_elem->colind & 3) == 1) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } else if ((cur_elem->colind & 3) == 2) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 2) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[2];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 3) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[3];
        } 
#elif INT32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif INT64
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#elif FP32
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#elif FP64 
        mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
        acc += cur_elem->val * cache_x[0];
#else
        if ((cur_elem->colind & 1) == 0) {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + cur_elem->colind * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[0];
        } else {
            mram_read((__mram_ptr void const *) (mram_base_addr_x + (cur_elem->colind - 1) * sizeof(val_dt)), cache_x, 8);
            acc += cur_elem->val * cache_x[1];
        } 
#endif

        // Get next nnz
        cur_elem = seqread_get(cur_elem, sizeof(struct elem_t), &sr_elem);
    }

    // Store output for the last output vector element in MRAM
    // Acquire the lock for MRAM writes
    diff = prev_row - tstart_row;
#if INT8
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff >> 3;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    if ((diff & 7) == 0) {
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 1) {
        diff -= 1; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[1] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 2) {
        diff -= 2; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[2] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 3) {
        diff -= 3; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[3] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 4) {
        diff -= 4; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[4] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 5) {
        diff -= 5; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[5] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 7) == 6) {
        diff -= 6; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[6] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else {
        diff -= 7; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + diff); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[7] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    }
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT16
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff >> 2;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    if ((diff & 3) == 0) {
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 3) == 1) {
        diff -= 1; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[1] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else if ((diff & 3) == 2) {
        diff -= 2; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[2] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else {
        diff -= 3; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 1)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[3] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    }
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT32
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff >> 1;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    if ((diff & 1) == 0) {
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8); 
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else {
        diff -= 1; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[1] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    }
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#elif INT64
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
    mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
    cache_y[0] += acc;
    mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#elif FP32
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff >> 1;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    if ((diff & 1) == 0) {
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else {
        diff -= 1; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[1] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    }
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#elif FP64 
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 3)); 
    mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
    cache_y[0] += acc;
    mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#else
#if CGLOCK
    mutex_lock(my_mutex);
#elif FGLOCK
    uint32_t lock_id = diff >> 1;
    lock_id = lock_id & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    if ((diff & 1) == 0) {
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[0] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } else {
        diff -= 1; 
        mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (diff << 2)); 
        mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
        cache_y[1] += acc;
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    }
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

#endif


EXIT: 


    return 0;
}
#endif
