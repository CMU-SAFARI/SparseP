#ifndef _COMMON_H_
#define _COMMON_H_

/* Structures used by both the host and the dpu to communicate information */
typedef struct {
    uint32_t nrows;
    uint32_t max_rows;
    uint32_t max_nnz_ind;
    uint32_t tcols;
    uint32_t nnz_pad;
    uint32_t nnz_offset;
    uint32_t start_row[NR_TASKLETS];
    uint32_t rows_per_tasklet[NR_TASKLETS];
} dpu_arguments_t;

#endif
