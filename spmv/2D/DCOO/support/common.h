#ifndef _COMMON_H_
#define _COMMON_H_


/* Structures used by both the host and the dpu to communicate information */
typedef struct {
    uint32_t max_rows_per_dpu;
    uint32_t tcols;
    uint32_t start_nnz[NR_TASKLETS];
    uint32_t nnz_per_tasklet[NR_TASKLETS];
} dpu_arguments_t;

#endif
