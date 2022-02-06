/**
 * cgiannoula: christina.giann@gmail.com
 * Christina Giannoula
 * Sparse matrix vector multiplication with multiple tasklets.
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


#if INT64 || FP64
//#define SEQREAD_CACHE_SIZE 128
#endif

int main_kernel_small_blocks(uint32_t tasklet_id) {

    // Load parameters
    uint32_t block_rows = DPU_INPUT_ARGUMENTS.block_rows;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row;
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block = DPU_INPUT_ARGUMENTS.start_block[0];
    uint32_t start_block = DPU_INPUT_ARGUMENTS.start_block[tasklet_id];
    uint32_t blocks_per_tasklet = DPU_INPUT_ARGUMENTS.blocks_per_tasklet[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_bind = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_bind + (max_blocks * sizeof(struct bind_t)));
    uint32_t mram_temp_addr_y;

    uint32_t row_size = 8;
    uint32_t col_size = 8;
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);   
    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    val_dt acc;
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < 64; i++) {
            cache_acc[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = (block_rows + 1) >> 1;
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), 8);
            mram_temp_addr_y += 8;
        }

        // Allocate cache memory to temporarily store results for the output vector elements
        sync_rptr = mem_alloc(NR_TASKLETS * 8);
        sync_values = mem_alloc(NR_TASKLETS * 2 * row_block_size);
        iter = NR_TASKLETS * 2 * row_block_size;
        for(i=0; i < iter; i++) {
            sync_values[i] = 0;
        }
    }
    barrier_wait(&my_barrier);

    // If there is no work, return
    if (blocks_per_tasklet == 0)
        goto EXIT1;

    // Initialize sequential reader for browptr, bcolind (indexes)
    mram_base_addr_bind += ((start_block - global_start_block) * sizeof(struct bind_t)); 
    seqreader_buffer_t cache_bind = seqread_alloc();
    seqreader_t sr_bind;
    struct bind_t *current_bind = seqread_init(cache_bind, (__mram_ptr void *) mram_base_addr_bind, &sr_bind);
    uint32_t prev_block_row = current_bind->rowind;

    // Initialize cache for bvalues
    mram_base_addr_val += ((start_block - global_start_block) * block_size); 
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
    mram_base_addr_val += block_size;

    uint32_t row_bound = 2; 
    uint32_t row_indx = 0; 
    uint32_t sync_base = tasklet_id * row_bound * row_block_size;

    // Find indexes for critical elements shared among multiple tasklets
    diff = prev_block_row - global_start_block_row;
    row_indx = (diff & 1);
    sync_rptr[tasklet_id] = diff - (diff & 1);
    if (blocks_per_tasklet == 0)
        sync_rptr[tasklet_id] = 0;


    // Initialize help variables
    uint32_t r, c;
    for(r = 0; r < 8; r++) {
        cache_acc[r] = 0;
    }

    // SpMV for critical elements
    // Iterate over blocks
    for (i=0; i < blocks_per_tasklet; i++) {
        // Temporarily store values of the critical elements  to sync_values cache
        if (current_bind->rowind != prev_block_row) {
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0)
                    sync_values[sync_base + row_indx * row_block_size + r] = cache_acc[r];
            }
            for(r = 0; r < row_block_size; r++) {
                cache_acc[r] = 0;
            }

            // If the next output vector subset of elements is computed by one tasklet (non critical elements),
            // break this loop and move to the main loop to directly store (final) results in MRAM
            row_indx += (current_bind->rowind - prev_block_row);
            if(row_indx >= row_bound) {
                prev_block_row = current_bind->rowind;
                break;
            }

            prev_block_row = current_bind->rowind;
        }

        // For eacn non-zero element sub-block: 1. get input vector value, 2. multiply and add
        if ((current_bind->colind & 1) == 1) {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind - 1) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c + col_block_size];
                }
            }
        } else {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
                }
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;

    }

    // Store output for the last critical output vector elements in sync_values array
    if (i == blocks_per_tasklet && row_indx < row_bound) {
        for(r = 0; r < row_block_size; r++) {
            if (cache_acc[r] != 0)
                sync_values[sync_base + row_indx * row_block_size + r] += cache_acc[r];
        }
    }    

    // SpMV
    // Iterate over the non-zero blocks
    for (i; i < blocks_per_tasklet; i++) {
        // If all non-zero blocks of the same block row have been traversed, write the final values for the output vector elements in MRAM
        if (current_bind->rowind != prev_block_row) {
            diff = prev_block_row - global_start_block_row;
            if((diff & 1) == 1)
                mram_temp_addr_y = (mram_base_addr_y + ((diff - 1) * row_block_size * sizeof(val_dt))); 
            else
                mram_temp_addr_y = (mram_base_addr_y + (diff * row_block_size * sizeof(val_dt))); 

            // Store the final values for the output vector element in MRAM  (block-row granularity)
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, 8);
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0) {
                    if ((diff & 1) == 1)
                        cache_y[r + row_block_size] += cache_acc[r];
                    else
                        cache_y[r] += cache_acc[r];
                }
            }
            for(r = 0; r < row_block_size; r++) {
                cache_acc[r] = 0;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);

            prev_block_row = current_bind->rowind;
        }

        // For each non-zero block: 1. get input vector value, 2. multiply and add
        // Note the 8-byte alignment needed
        if ((current_bind->colind & 1) == 1) {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind - 1) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c + col_block_size];
                }
            }
        } else {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
                }
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;

    }

    // Store the final values for the output vector element in MRAM  (block-row granularity)
    if (row_indx >= row_bound) {
        diff = prev_block_row - global_start_block_row;
        if((diff & 1) == 1)
            mram_temp_addr_y = (mram_base_addr_y + ((diff - 1) * row_block_size * sizeof(val_dt))); 
        else
            mram_temp_addr_y = (mram_base_addr_y + (diff * row_block_size * sizeof(val_dt))); 


        mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, 8);
        for(r = 0; r < row_block_size; r++) {
            if (cache_acc[r] != 0) {
                if ((diff & 1) == 1)
                    cache_y[r + row_block_size] += cache_acc[r];
                else
                    cache_y[r] += cache_acc[r];
            }
        }
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
    } 

EXIT1:

    // Wait for all tasklets to finish
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        // One tasklet merges the partial results that are temporarily stored in sync_values cache
        // Based on their indexes stored in sync_rptr cache one tasklet writes the final output vector element values to MRAM
        uint32_t t = 0;
        for(i = 0; i < NR_TASKLETS; i++) {
            // For each tasklet find the partial results that it has created and merge them in MRAM
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] * row_block_size * sizeof(val_dt))); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, 8);
            for(r=0; r < 2 * row_block_size; r++) {
                cache_y[r] += sync_values[t]; 
                t++;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
        }
    }

    return 0;
}


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

#if INT8
    // If int8 data type and 4x4 block size, use a specialized kernel function to ensure 8-byte alignment
    if (DPU_INPUT_ARGUMENTS.row_block_size == 4) {
        main_kernel_small_blocks(tasklet_id);
        goto EXIT;
    }
#endif

#if INT16
    // If int16 data type and 2x2 block size, use a specialized kernel function to ensure 8-byte alignment
    if (DPU_INPUT_ARGUMENTS.row_block_size == 2) {
        main_kernel_small_blocks(tasklet_id);
        goto EXIT;
    }
#endif

    // Load parameters
    uint32_t block_rows = DPU_INPUT_ARGUMENTS.block_rows;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row;
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block = DPU_INPUT_ARGUMENTS.start_block[0];
    uint32_t start_block = DPU_INPUT_ARGUMENTS.start_block[tasklet_id];
    uint32_t blocks_per_tasklet = DPU_INPUT_ARGUMENTS.blocks_per_tasklet[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_bind = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_bind + (max_blocks * sizeof(struct bind_t)));
    uint32_t mram_temp_addr_y;

    uint32_t row_size = row_block_size * sizeof(val_dt);
    uint32_t col_size = col_block_size * sizeof(val_dt);
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);   
    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    // Help variables
    val_dt acc;
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < row_block_size; i++) {
            cache_acc[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = block_rows; 
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), row_block_size * sizeof(val_dt));
            mram_temp_addr_y += (row_size);
        }

        // Allocate cache memory to temporarily store results for the output vector elements
        sync_rptr = mem_alloc(NR_TASKLETS * 8);
        sync_values = mem_alloc(NR_TASKLETS * row_block_size * sizeof(val_dt));
        iter = NR_TASKLETS * row_block_size;
        for(i=0; i < iter; i++) {
            sync_values[i] = 0;
        }
    }
    barrier_wait(&my_barrier);

    // Initialize sequential reader for browptr, bcolind (indexes)
    mram_base_addr_bind += ((start_block - global_start_block) * sizeof(struct bind_t)); 
    seqreader_buffer_t cache_bind = seqread_alloc();
    seqreader_t sr_bind;
    struct bind_t *current_bind = seqread_init(cache_bind, (__mram_ptr void *) mram_base_addr_bind, &sr_bind);
    uint32_t prev_block_row = current_bind->rowind;

    // Initialize cache for bvalues
    mram_base_addr_val += ((start_block - global_start_block) * block_size); 
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
    mram_base_addr_val += block_size;

    // Find indexes of critical elements shared among multiple tasklets
    uint32_t sync_bool = 0;
    uint32_t sync_base = tasklet_id * row_block_size;
    diff = prev_block_row - global_start_block_row;
    sync_rptr[tasklet_id] = diff;
    if (blocks_per_tasklet == 0)
        sync_rptr[tasklet_id] = 0;


    // Initialize help variables
    uint32_t r, c;
    for(r = 0; r < row_block_size; r++) {
        cache_acc[r] = 0;
    }


    // SpMV for critical elements
    // Iterate over blocks
    for (i=0; i < blocks_per_tasklet; i++) {
        // Temporarily store values of the critical elements  to sync_values cache
        if (current_bind->rowind != prev_block_row) {
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0)
                    sync_values[sync_base + r] = cache_acc[r];
                cache_acc[r] = 0;
            }

            // If the next output vector subset of elements is computed by one tasklet (non critical elements),
            // break this loop and move to the main loop to directly store (final) results in MRAM
            prev_block_row = current_bind->rowind;
            break;
        }

        // For eacn non-zero element sub-block: 1. get input vector value, 2. multiply and add
        mram_read((__mram_ptr void const *) (mram_base_addr_x + ((current_bind->colind) * col_size)), cache_x, col_size);
        for(r = 0; r < row_block_size; r++) {
            acc = 0;
            for(c = 0; c < col_block_size; c++) {
                if (cache_val[r * col_block_size + c]  != 0)
                    cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;

    }

    // Store output for the last critical output vector elements in sync_values array
    if (i > 0 && i == blocks_per_tasklet) {
        for(r = 0; r < row_block_size; r++) {
            if (cache_acc[r] != 0)
                sync_values[sync_base + r] = cache_acc[r];
        }
        sync_bool = 1;
    }


    // SpMV
    // Iterate over the non-zero blocks
    for (i; i < blocks_per_tasklet; i++) {
        // If all non-zero blocks of the same block row have been traversed, write the final values for the output vector elements in MRAM
        if (current_bind->rowind != prev_block_row) {
            diff = prev_block_row - global_start_block_row;

            // Store the final values for the output vector element in MRAM  (block-row granularity)
            mram_temp_addr_y = (mram_base_addr_y + (diff * row_size)); 
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, row_size);
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0)
                    cache_y[r] += cache_acc[r];
                cache_acc[r] = 0;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), row_size);

            prev_block_row = current_bind->rowind;
        }

        // For each non-zero block: 1. get input vector value, 2. multiply and add
        mram_read((__mram_ptr void const *) (mram_base_addr_x + ((current_bind->colind) * col_size)), cache_x, col_size);
        for(r = 0; r < row_block_size; r++) {
            acc = 0;
            for(c = 0; c < col_block_size; c++) {
                if (cache_val[r * col_block_size + c]  != 0)
                    cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;
    }

    // Store the final values for the output vector element in MRAM  (block-row granularity)
    if (sync_bool == 0 && blocks_per_tasklet > 0) {
        diff = prev_block_row - global_start_block_row;

        mram_temp_addr_y = (mram_base_addr_y + (diff * row_size)); 
        mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, row_size);
        for(r = 0; r < row_block_size; r++) {
            if (cache_acc[r] != 0)
                cache_y[r] += cache_acc[r];
        }
        mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), row_size);
    }

    // Wait for all tasklets to finish
    barrier_wait(&my_barrier);
    if (tasklet_id == 0) {
        // One tasklet merges the partial results that are temporarily stored in sync_values cache
        // Based on their indexes stored in sync_rptr cache one tasklet writes the final output vector element values to MRAM
        uint32_t t = 0;
        for(i = 0; i < NR_TASKLETS; i++) {
            // For each tasklet find the partial results that it has created and merge them in MRAM
            mram_temp_addr_y = (uint32_t) (mram_base_addr_y + (sync_rptr[i] * row_size)); 
            mram_read((__mram_ptr void *) (mram_temp_addr_y), cache_y, row_size);
            for(r=0; r < row_block_size; r++) {
                cache_y[r] += sync_values[t]; 
                t++;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), row_size);
        }
    }

EXIT:

    return 0;
}


#else
// LOCK-based implementations (i.e., CGLOCK and FGLOCK)


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
#if CGLOCK
MUTEX_INIT(my_mutex);
#elif FGLOCK
mutex_id_t my_mutex[32];
#endif

#if INT64 || FP64
//#define SEQREAD_CACHE_SIZE 128
#endif

int main_kernel_small_blocks(uint32_t tasklet_id) {

    // Load parameters
    uint32_t block_rows = DPU_INPUT_ARGUMENTS.block_rows;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row;
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block = DPU_INPUT_ARGUMENTS.start_block[0];
    uint32_t start_block = DPU_INPUT_ARGUMENTS.start_block[tasklet_id];
    uint32_t blocks_per_tasklet = DPU_INPUT_ARGUMENTS.blocks_per_tasklet[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_bind = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_bind + (max_blocks * sizeof(struct bind_t)));
    uint32_t mram_temp_addr_y;

    uint32_t row_size = 8;
    uint32_t col_size = 8;
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);   
    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    // Help variables
    val_dt acc;
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < 64; i++) {
            cache_acc[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = (block_rows + 1) >> 1; 
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), 8);
            mram_temp_addr_y += 8;
        }
    }
    barrier_wait(&my_barrier);

    // If there is no work, return
    if (blocks_per_tasklet == 0)
        goto EXIT1;

    // Initialize sequential reader for browptr, bcolind (indexes)
    mram_base_addr_bind += ((start_block - global_start_block) * sizeof(struct bind_t)); 
    seqreader_buffer_t cache_bind = seqread_alloc();
    seqreader_t sr_bind;
    struct bind_t *current_bind = seqread_init(cache_bind, (__mram_ptr void *) mram_base_addr_bind, &sr_bind);
    uint32_t prev_block_row = current_bind->rowind;

    // Initialize cache for bvalues
    mram_base_addr_val += ((start_block - global_start_block) * block_size); 
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
    mram_base_addr_val += block_size;

    // Initialize help caches
    uint32_t r, c;
    for(r = 0; r < 8; r++) {
        cache_acc[r] = 0;
    }

    // SpMV
    // Iterate over block rows 
    for (i=0; i < blocks_per_tasklet; i++) {
        // If all non-zero blocks of the same block row have been traversed, write the final values for the output vector elements in MRAM
        if (current_bind->rowind != prev_block_row) {
            diff = prev_block_row - global_start_block_row;
            if((diff & 1) == 1) // 8-byte alignement needed
                mram_temp_addr_y = (mram_base_addr_y + ((diff - 1) * row_block_size * sizeof(val_dt))); 
            else
                mram_temp_addr_y = (mram_base_addr_y + (diff * row_block_size * sizeof(val_dt))); 

            // Store the final values for the output vector element in MRAM  (block-row granularity)
            // Using locks
#if CGLOCK
            mutex_lock(my_mutex); 
#elif FGLOCK
            uint32_t lock_id = diff & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, 8);
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0) {
                    if ((diff & 1) == 1)
                        cache_y[r + row_block_size] += cache_acc[r];
                    else
                        cache_y[r] += cache_acc[r];
                }
            }
            for(r = 0; r < 8; r++) {
                cache_acc[r] = 0;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif
            prev_block_row = current_bind->rowind;
        }

        // For each non-zero block: 1. get input vector value, 2. multiply and add
        if ((current_bind->colind & 1) == 1) {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind - 1) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c + col_block_size];
                }
            }
        } else {
            mram_temp_addr_y = mram_base_addr_x + ((current_bind->colind) * col_block_size * sizeof(val_dt));
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_x, 8);
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c]  != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
                }
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;

    }

    // Store the final values for the output vector element in MRAM  (block-row granularity)
    diff = prev_block_row - global_start_block_row;
    if((diff & 1) == 1)
        mram_temp_addr_y = (mram_base_addr_y + ((diff - 1) * row_block_size * sizeof(val_dt))); 
    else
        mram_temp_addr_y = (mram_base_addr_y + (diff * row_block_size * sizeof(val_dt))); 

#if CGLOCK
    mutex_lock(my_mutex); 
#elif FGLOCK
    uint32_t lock_id = diff & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, 8);
    for(r = 0; r < row_block_size; r++) {
        if (cache_acc[r] != 0) {
            if ((diff & 1) == 1)
                cache_y[r + row_block_size] += cache_acc[r];
            else
                cache_y[r] += cache_acc[r];
        }
    }
    mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), 8);
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif


EXIT1:

    return 0;
}

/**
 * @brief main function executed by each tasklet
 */
int main() {
    uint32_t tasklet_id = me();

    if (tasklet_id == 0){ 
        mem_reset(); // Reset the heap
    }

    //Barrier
    barrier_wait(&my_barrier);

#if INT8
    // If int8 data type and 4x4 block size, use a specialized kernel function to ensure 8-byte alignment
    if (DPU_INPUT_ARGUMENTS.row_block_size == 4) {
        main_kernel_small_blocks(tasklet_id);
        goto EXIT;
    }
#endif

#if INT16
    // If int16 data type and 2x2 block size, use a specialized kernel function to ensure 8-byte alignment
    if (DPU_INPUT_ARGUMENTS.row_block_size == 2) {
        main_kernel_small_blocks(tasklet_id);
        goto EXIT;
    }
#endif

    // Load parameters
    uint32_t block_rows = DPU_INPUT_ARGUMENTS.block_rows;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row;
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block = DPU_INPUT_ARGUMENTS.start_block[0];
    uint32_t start_block = DPU_INPUT_ARGUMENTS.start_block[tasklet_id];
    uint32_t blocks_per_tasklet = DPU_INPUT_ARGUMENTS.blocks_per_tasklet[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_bind = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_val = (uint32_t) (mram_base_addr_bind + (max_blocks * sizeof(struct bind_t)));
    uint32_t mram_temp_addr_y;

    uint32_t row_size = row_block_size * sizeof(val_dt);
    uint32_t col_size = col_block_size * sizeof(val_dt);
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);   

    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    // Help variables
    val_dt acc;
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < row_block_size; i++) {
            cache_acc[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = block_rows; 
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), row_block_size * sizeof(val_dt));
            mram_temp_addr_y += (row_size);
        }
    }
    barrier_wait(&my_barrier);

    // If there is no work, return
    if (blocks_per_tasklet == 0)
        goto EXIT;

    // Find offsets per tasklet for browptr, bcolind (indexes)
    mram_base_addr_bind += ((start_block - global_start_block) * sizeof(struct bind_t)); 
    seqreader_buffer_t cache_bind = seqread_alloc();
    seqreader_t sr_bind;
    struct bind_t *current_bind = seqread_init(cache_bind, (__mram_ptr void *) mram_base_addr_bind, &sr_bind);
    uint32_t prev_block_row = current_bind->rowind;

    // Initialize cache for bvalues
    mram_base_addr_val += ((start_block - global_start_block) * block_size); 
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
    mram_base_addr_val += block_size;

    // Initialize cache_acc
    uint32_t r, c;
    for(r = 0; r < row_block_size; r++) {
        cache_acc[r] = 0;
    }

    // SpMV
    // Iterate over block rows 
    for (i=0; i < blocks_per_tasklet; i++) {
        // If all non-zero blocks of the same block row have been traversed, write the final values for the output vector elements in MRAM
        if (current_bind->rowind != prev_block_row) {
            diff = prev_block_row - global_start_block_row;
            mram_temp_addr_y = (mram_base_addr_y + (diff * row_size)); 

            // Store the final values for the output vector element in MRAM  (block-row granularity)
            // Using locks
#if CGLOCK
            mutex_lock(my_mutex); 
#elif FGLOCK
            uint32_t lock_id = diff & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, row_size);
            for(r = 0; r < row_block_size; r++) {
                if (cache_acc[r] != 0)
                    cache_y[r] += cache_acc[r];
                cache_acc[r] = 0;
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), row_size);
#if CGLOCK
            mutex_unlock(my_mutex);
#elif FGLOCK
            mutex_unlock(my_mutex[lock_id]);    
#endif
            prev_block_row = current_bind->rowind;
        }

        // For each non-zero block: 1. get input vector value, 2. multiply and add
        mram_read((__mram_ptr void const *) (mram_base_addr_x + ((current_bind->colind) * col_size)), cache_x, col_size);
        for(r = 0; r < row_block_size; r++) {
            acc = 0;
            for(c = 0; c < col_block_size; c++) {
                if (cache_val[r * col_block_size + c]  != 0)
                    cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
            }
        }

        // Get the next non-zero block
        current_bind = seqread_get(current_bind, sizeof(*current_bind), &sr_bind);
        mram_read((__mram_ptr void const *) (mram_base_addr_val), cache_val, block_size);
        mram_base_addr_val += block_size;

    }

    // Store the final values for the output vector element in MRAM  (block-row granularity)
    diff = prev_block_row - global_start_block_row;
    mram_temp_addr_y = (mram_base_addr_y + (diff * row_size)); 

#if CGLOCK
    mutex_lock(my_mutex); 
#elif FGLOCK
    uint32_t lock_id = diff & 0x0000001F;
    mutex_lock(my_mutex[lock_id]);    
#endif
    mram_read((__mram_ptr void const *) (mram_temp_addr_y), cache_y, row_size);
    for(r = 0; r < row_block_size; r++) {
        if (cache_acc[r] != 0)
            cache_y[r] += cache_acc[r];
    }
    mram_write(cache_y, (__mram_ptr void *) (mram_temp_addr_y), row_size);
#if CGLOCK
    mutex_unlock(my_mutex);
#elif FGLOCK
    mutex_unlock(my_mutex[lock_id]);    
#endif

EXIT:


    return 0;
}
#endif
