#ifndef _COMMON_H_
#define _COMMON_H_

/* Structures used by both the host and the dpu to communicate information */
typedef struct {
    uint32_t block_rows;
    uint32_t start_block_row;
    uint32_t max_block_rows;
    uint32_t row_block_size;
    uint32_t col_block_size;
    uint32_t tcols;
    uint32_t max_blocks;
    uint32_t dummy;
    uint32_t start_block[NR_TASKLETS];
    uint32_t blocks_per_tasklet[NR_TASKLETS];
} dpu_arguments_t;

#endif
