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
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        uint64_t *temp = mem_alloc(2048);
        uint32_t iter = 256;
        for(i=0; i < iter; i++) {
            temp[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t init_bytes = (((block_rows +1) >> 1) * 8); 
        iter = (init_bytes >> 11); 
        uint32_t acc_init = 0;
        for(i=0; i < iter; i++) {
            mram_write(temp, (__mram_ptr void *) (mram_temp_addr_y), 2048);
            mram_temp_addr_y += 2048;
            acc_init += 2048;
        }

        for(i=acc_init; i < init_bytes; i+=8) {
            mram_write(temp, (__mram_ptr void *) (mram_temp_addr_y), 8);
            mram_temp_addr_y += 8;
        }
        mem_reset(); // Reset the heap

    }
    barrier_wait(&my_barrier);

    // If there is no work, return
    if (blocks_per_tasklet == 0)
        goto EXIT1;

    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    val_dt acc;

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
            if((diff & 1) == 1)
                mram_temp_addr_y = (mram_base_addr_y + ((diff - 1) * row_block_size * sizeof(val_dt))); 
            else
                mram_temp_addr_y = (mram_base_addr_y + (diff * row_block_size * sizeof(val_dt))); 

            // Store the final values for the output vector element in MRAM  (block-row granularity)
            // Using locks
#if CGLOCK
            mutex_lock(my_mutex); 
#else
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
#else
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
#else
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
#else
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
    uint32_t i, diff;

    // Use cache_acc cache to initialize the output vector elements in MRAM with zeros
    if(tasklet_id == 0) { 
        uint64_t *temp = mem_alloc(2048);
        uint32_t iter = 256;
        for(i=0; i < iter; i++) {
            temp[i] = 0;
        }

        mram_temp_addr_y = (uint32_t) (DPU_MRAM_HEAP_POINTER);
        uint32_t init_bytes = (block_rows * row_block_size * sizeof(val_dt)); 
        iter = (init_bytes >> 11); 
        uint32_t acc_init = 0;
        for(i=0; i < iter; i++) {
            mram_write(temp, (__mram_ptr void *) (mram_temp_addr_y), 2048);
            mram_temp_addr_y += 2048;
            acc_init += 2048;
        }
        for(i=acc_init; i < init_bytes; i+=8) {
            mram_write(temp, (__mram_ptr void *) (mram_temp_addr_y), 8);
            mram_temp_addr_y += 8;
        }
        mem_reset(); // Reset the heap

    }
    barrier_wait(&my_barrier);

    // If there is no work, return
    if (blocks_per_tasklet == 0)
        goto EXIT;

    // Initialize help cache to temporarily store results
    val_dt *cache_acc = mem_alloc(row_size);
    // Initialize input vector cache
    val_dt *cache_x = mem_alloc(col_size);
    // Initialize output vector cache
    val_dt *cache_y = mem_alloc(row_size);
    val_dt acc;

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
#else
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
#else
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
#else
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
#else
    mutex_unlock(my_mutex[lock_id]);    
#endif

EXIT:

    return 0;
}
