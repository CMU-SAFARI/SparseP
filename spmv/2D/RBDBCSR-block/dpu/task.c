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
#include <mutex.h>
#include <seqread.h>

#include "../support/common.h"
#include "../support/utils.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;


// Global Variables
// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);
#if CGLOCK
MUTEX_INIT(my_mutex);
#else
mutex_id_t my_mutex[32];
#endif


#if INT64 || FP64
//#define SEQREAD_CACHE_SIZE 128
#endif

int main_kernel_small_blocks(uint32_t tasklet_id) {

    // Load parameters
    uint32_t block_rows = DPU_INPUT_ARGUMENTS.block_rows;
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row[0];
    uint32_t start_block_row = DPU_INPUT_ARGUMENTS.start_block_row[tasklet_id];
    uint32_t end_block_row = DPU_INPUT_ARGUMENTS.end_block_row[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_browptr = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_bcolind = (uint32_t) (mram_base_addr_browptr + (max_block_rows * sizeof(uint32_t)));
    uint32_t mram_base_addr_values = (uint32_t) (mram_base_addr_bcolind + (max_blocks * sizeof(uint32_t)));

    // Initialize help cache memory and find the start global block row of the DPU
    uint32_t *cache_temp = mem_alloc(8); 
    mram_read((__mram_ptr void const *) (mram_base_addr_browptr), (void *) (cache_temp), 8);
    uint32_t global_block_offset = cache_temp[0]; 

    // Initialize sequential reader for browptr
    mram_base_addr_browptr += ((start_block_row - global_start_block_row) * sizeof(uint32_t));
    seqreader_buffer_t cache_browptr = seqread_alloc();
    seqreader_t sr_browptr;
    uint32_t *current_row = seqread_init(cache_browptr, (__mram_ptr void *) mram_base_addr_browptr, &sr_browptr);
    uint32_t prev_row = *current_row;

    // Find offsets per tasklet for bcolind and bvalues arrays
    mram_base_addr_bcolind += ((prev_row - global_block_offset) * sizeof(uint32_t));
    mram_base_addr_values += ((prev_row - global_block_offset) * row_block_size * col_block_size * sizeof(val_dt));

    // Initialize help structures
    // write_bound, write_indx are used to temporarily store results in WRAM for the output vector elements,
    // such that to ensure that MRAM writes are performed in 8-byte granularity
    uint32_t write_indx = 0;
    uint32_t write_bound = 2;
    if (((start_block_row - global_start_block_row) & 1) == 1) {
        mram_base_addr_y += ((start_block_row - global_start_block_row - 1) * row_block_size  * sizeof(val_dt)); 
        write_indx = 1;
    } else {
        mram_base_addr_y += ((start_block_row - global_start_block_row) * row_block_size  * sizeof(val_dt)); 
    }

    // Initialize sequential reader for bcolind
    seqreader_buffer_t cache_bcolind = seqread_alloc();
    seqreader_t sr_bcolind;
    uint32_t *current_bcolind = seqread_init(cache_bcolind, (__mram_ptr void *) mram_base_addr_bcolind, &sr_bcolind);

    // Initialize cache for bvalues
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_values), cache_val, block_size);
    mram_base_addr_values += block_size;

    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(8);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(8);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(8);
    // Help variables
    val_dt acc;
    uint32_t i, j, curr_block_start, curr_block_end;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < 8; i++) {
            cache_acc[i] = 0;
        }

        uint32_t mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = (block_rows + 1) >> 1; 
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), 8);
            mram_temp_addr_y += 8;
        }
    }
    barrier_wait(&my_barrier);


    // Initialized cache_acc, cache_r memory
    uint32_t r, c;
    for(r = 0; r < 8; r++) {
        cache_acc[r] = 0;
        cache_y[r] = 0;
    }

    // SpMV
    // Iterate over block rows 
    for (i=start_block_row; i < end_block_row; i++) {
        // Read pointer of the browptr array
        current_row = seqread_get(current_row, sizeof(*current_row), &sr_browptr);
        curr_block_start = prev_row;
        curr_block_end = *current_row;

        // Iterate over the non-zero blocks of the same block row
        for(j = curr_block_start; j < curr_block_end; j++) {
            // Read the corresponding subset of elements of the input vector
            if ((*current_bcolind & 1) == 1) {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_bcolind - 1) * col_block_size * sizeof(val_dt))), cache_x, 8);
                // For each non-zero block: 1. get input vector value, 2. multiply and add
                for(r = 0; r < row_block_size; r++) {
                    acc = 0;
                    for(c = 0; c < col_block_size; c++) {
                        if (cache_val[r * col_block_size + c] != 0)
                            cache_acc[r + write_indx * row_block_size] += cache_val[r * col_block_size + c] * cache_x[c + col_block_size];
                    }
                }

            } else {
                mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_bcolind) * col_block_size * sizeof(val_dt))), cache_x, 8);
                // For each non-zero block: 1. get input vector value, 2. multiply and add
                for(r = 0; r < row_block_size; r++) {
                    acc = 0;
                    for(c = 0; c < col_block_size; c++) {
                        if (cache_val[r * col_block_size + c] != 0)
                            cache_acc[r + write_indx * row_block_size] += cache_val[r * col_block_size + c] * cache_x[c];
                    }
                }
            }

            // Get the next non-zero block
            current_bcolind = seqread_get(current_bcolind, sizeof(*current_bcolind), &sr_bcolind);
            mram_read((__mram_ptr void const *) (mram_base_addr_values), cache_val, block_size);
            mram_base_addr_values += block_size;
        }


        if (write_indx != write_bound - 1){
            // Temporarily store final values in cache memory to ensure 8-byte MRAM writes
            write_indx++;
        } else {
            // Store the final values for the output vector element in MRAM  (block-row granularity)
#if CGLOCK
            mutex_lock(my_mutex);
#else
            uint32_t lock_id = i & 0x0000001F;
            mutex_lock(my_mutex[lock_id]);    
#endif
            mram_read((__mram_ptr void *) (mram_base_addr_y), cache_y, 8);
            for(r = 0; r < 8; r++) {
                cache_y[r] += cache_acc[r];
            }
            mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
#if CGLOCK
            mutex_unlock(my_mutex);
#else
            mutex_unlock(my_mutex[lock_id]);    
#endif
            // Initialize cache_acc with zeros
            for(r = 0; r < 8; r++) {
                cache_acc[r] = 0;
            }
            // Move to the next subset of elements of the output vector (block-row granularity)
            mram_base_addr_y += 8;
            write_indx = 0;
        }

        prev_row = *current_row;
    }

    if (write_indx != 0) {
        // Store the last final values for the output vector element in MRAM  (block-row granularity)
#if CGLOCK
        mutex_lock(my_mutex);
#else
        uint32_t lock_id = i & 0x0000001F;
        mutex_lock(my_mutex[lock_id]);    
#endif
        mram_read((__mram_ptr void *) (mram_base_addr_y), cache_y, 8);
        for(r = 0; r < 8; r++) {
            cache_y[r] += cache_acc[r];
        }
        mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), 8);
#if CGLOCK
        mutex_unlock(my_mutex);
#else
        mutex_unlock(my_mutex[lock_id]);    
#endif
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
    uint32_t max_block_rows = DPU_INPUT_ARGUMENTS.max_block_rows;
    uint32_t row_block_size = DPU_INPUT_ARGUMENTS.row_block_size;
    uint32_t col_block_size = DPU_INPUT_ARGUMENTS.col_block_size;
    uint32_t tcols = DPU_INPUT_ARGUMENTS.tcols;
    uint32_t max_blocks = DPU_INPUT_ARGUMENTS.max_blocks;
    uint32_t global_start_block_row = DPU_INPUT_ARGUMENTS.start_block_row[0];
    uint32_t start_block_row = DPU_INPUT_ARGUMENTS.start_block_row[tasklet_id];
    uint32_t end_block_row = DPU_INPUT_ARGUMENTS.end_block_row[tasklet_id];

    // Find start addresses in MRAM
    uint32_t mram_base_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
    uint32_t mram_base_addr_x = (uint32_t) (DPU_MRAM_HEAP_POINTER + (max_block_rows * row_block_size * sizeof(val_dt)));
    uint32_t mram_base_addr_browptr = (uint32_t) (mram_base_addr_x + (tcols * sizeof(val_dt)));
    uint32_t mram_base_addr_bcolind = (uint32_t) (mram_base_addr_browptr + (max_block_rows * sizeof(uint32_t)));
    uint32_t mram_base_addr_values = (uint32_t) (mram_base_addr_bcolind + (max_blocks * sizeof(uint32_t)));

    // Initialize help cache memory and find the start global block row of the DPU
    uint32_t *cache_temp = mem_alloc(8); 
    mram_read((__mram_ptr void const *) (mram_base_addr_browptr), (void *) (cache_temp), 8);
    uint32_t global_block_offset = cache_temp[0]; 

    // Initialize sequential reader for browptr
    mram_base_addr_browptr += ((start_block_row - global_start_block_row) * sizeof(uint32_t));
    seqreader_buffer_t cache_browptr = seqread_alloc();
    seqreader_t sr_browptr;
    uint32_t *current_row = seqread_init(cache_browptr, (__mram_ptr void *) mram_base_addr_browptr, &sr_browptr);
    uint32_t prev_row = *current_row;

    // Find offsets per tasklet for bcolind and bvalues arrays
    mram_base_addr_bcolind += ((prev_row - global_block_offset) * sizeof(uint32_t));
    mram_base_addr_values += ((prev_row - global_block_offset) * row_block_size * col_block_size * sizeof(val_dt));
    mram_base_addr_y += ((start_block_row - global_start_block_row) * row_block_size  * sizeof(val_dt)); 

    // Initialize sequential reader for bcolind
    seqreader_buffer_t cache_bcolind = seqread_alloc();
    seqreader_t sr_bcolind;
    uint32_t *current_bcolind = seqread_init(cache_bcolind, (__mram_ptr void *) mram_base_addr_bcolind, &sr_bcolind);

    // Initialize cache for bvalues
    uint32_t block_size = row_block_size * col_block_size * sizeof(val_dt);
    val_dt *cache_val = mem_alloc(block_size);
    mram_read((__mram_ptr void const *) (mram_base_addr_values), cache_val, block_size);
    mram_base_addr_values += block_size;
    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_block_size * sizeof(val_dt));
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_block_size * sizeof(val_dt));
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_block_size * sizeof(val_dt));
    // Help variables
    val_dt acc;
    uint32_t i, j, curr_block_start, curr_block_end;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        for(i=0; i < row_block_size; i++) {
            cache_acc[i] = 0;
        }

        uint32_t mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t iter = block_rows; 
        for(i=0; i < iter; i++) {
            mram_write(cache_acc, (__mram_ptr void *) (mram_temp_addr_y), row_block_size * sizeof(val_dt));
            mram_temp_addr_y += (row_block_size * sizeof(val_dt));
        }
    }
    barrier_wait(&my_barrier);


    // Initialized cache_acc memory
    uint32_t r, c;
    for(r = 0; r < row_block_size; r++) {
        cache_acc[r] = 0;
    }

    // SpMV
    // Iterate over block rows 
    for (i=start_block_row; i < end_block_row; i++) {
        // Read pointer of the browptr array
        current_row = seqread_get(current_row, sizeof(*current_row), &sr_browptr);
        curr_block_start = prev_row;
        curr_block_end = *current_row;

        // Iterate over the non-zero blocks of the same block row
        for(j = curr_block_start; j < curr_block_end; j++) {
            // Read the corresponding subset of elements of the input vector
            mram_read((__mram_ptr void const *) (mram_base_addr_x + ((*current_bcolind) * col_block_size * sizeof(val_dt))), cache_x, col_block_size * sizeof(val_dt));
            // For each non-zero block: 1. get input vector value, 2. multiply and add
            for(r = 0; r < row_block_size; r++) {
                acc = 0;
                for(c = 0; c < col_block_size; c++) {
                    if (cache_val[r * col_block_size + c] != 0)
                        cache_acc[r] += cache_val[r * col_block_size + c] * cache_x[c];
                }

            }

            // Get the next non-zero block
            current_bcolind = seqread_get(current_bcolind, sizeof(*current_bcolind), &sr_bcolind);
            mram_read((__mram_ptr void const *) (mram_base_addr_values), cache_val, block_size);
            mram_base_addr_values += block_size;
        }

        // Store the final values for the output vector element in MRAM  (block-row granularity)
        mram_read((__mram_ptr void const *) (mram_base_addr_y), cache_y, row_block_size * sizeof(val_dt));
        for(r = 0; r < row_block_size; r++) {
            if (cache_acc[r] != 0)
                cache_y[r] += cache_acc[r];
            cache_acc[r] = 0;
        }
        mram_write(cache_y, (__mram_ptr void *) (mram_base_addr_y), row_block_size * sizeof(val_dt));

        // Move to the next subset of elements of the output vector (block-row granularity)
        mram_base_addr_y += row_block_size * sizeof(val_dt);

        prev_row = *current_row;
    }

EXIT:
    return 0;
}
