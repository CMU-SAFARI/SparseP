/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 *
 * Partitioning and balancing across DPUs and tasklets
 */

#ifndef _PARTITION_H_
#define _PARTITION_H_

/**
 * @brief Specific information for data partitioning/balancing
 */
struct partition_info_t {
    uint32_t *row_split;
    uint32_t *row_split_tasklet;
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(uint32_t nr_of_dpus, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->row_split = (uint32_t *) malloc((nr_of_dpus + 1) * sizeof(uint32_t));
    part_info->row_split_tasklet = (uint32_t *) malloc((nr_of_tasklets + 2) * sizeof(uint32_t));

    return part_info;
}

/** 
 * @brief load-balance nnz at row granularity across DPUs
 */
void partition_by_nnz(struct CSRMatrix *csrMtx, struct partition_info_t *part_info, int nr_of_dpus) {

    if (nr_of_dpus == 1) {
        part_info->row_split[0] = 0;
        part_info->row_split[1] = csrMtx->nrows;
        return;
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = csrMtx->nnz;
    uint32_t nnz_per_split = nnz_cnt / nr_of_dpus;
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i;

    part_info->row_split[0] = row_start;
    for (i = 0; i < csrMtx->nrows; i++) {
        curr_nnz += csrMtx->rowptr[i+1] - csrMtx->rowptr[i];
        if (curr_nnz >= nnz_per_split) {
            row_start = i + 1;
            ++split_cnt;
            if (split_cnt <= nr_of_dpus)
                part_info->row_split[split_cnt] = row_start;
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_dpus) {
        part_info->row_split[++split_cnt] = csrMtx->nrows;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_dpus) {
        part_info->row_split[nr_of_dpus] = csrMtx->nrows;
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_dpus; i++) {
        part_info->row_split[i] = csrMtx->nrows;
    }

}

#if BLNC_TSKLT_ROW
/** 
 * @brief load-balance rows across tasklets
 */
void partition_tsklt_by_row(struct partition_info_t* part_info, int rows_per_dpu, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t granularity = 1;
    granularity = 8 / byte_dt;
    uint32_t chunks = rows_per_dpu / (granularity * nr_of_tasklets); 
    uint32_t rest_rows = rows_per_dpu % (granularity * nr_of_tasklets); 
    uint32_t rows_per_tasklet;
    uint32_t curr_row = 0;

    part_info->row_split_tasklet[0] = curr_row;
    for(unsigned int i=0; i < nr_of_tasklets; i++) {
        rows_per_tasklet = (granularity * chunks);
        if (i < rest_rows)
            rows_per_tasklet += granularity;
        curr_row += rows_per_tasklet;
        if (curr_row > rows_per_dpu)
            curr_row = rows_per_dpu;
        part_info->row_split_tasklet[i+1] = curr_row;
    }
}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief balance nnz in row granularity across tasklets
 */
void partition_tsklt_by_nnz(struct CSRMatrix *csrMtx, struct partition_info_t* part_info, int rows_per_dpu, int nnz_per_dpu, int prev_rows_dpu, int nr_of_tasklets) {

    // Compute the matrix splits.
    uint32_t granularity = 1;
    granularity = 8 / byte_dt;
    uint32_t nnz_per_split = nnz_per_dpu / nr_of_tasklets; 
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t t;

    part_info->row_split_tasklet[0] = row_start;
    for (t = 0; t < rows_per_dpu; t++) {
        curr_nnz += csrMtx->rowptr[prev_rows_dpu+t+1] - csrMtx->rowptr[prev_rows_dpu+t];
        if ((curr_nnz >= nnz_per_split) && ((t+1) % granularity == 0)) {
            row_start = t + 1;
            ++split_cnt;
            if (split_cnt <= nr_of_tasklets) {
                part_info->row_split_tasklet[split_cnt] = row_start;
            }
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_tasklets) {
        part_info->row_split_tasklet[++split_cnt] = rows_per_dpu;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->row_split_tasklet[nr_of_tasklets] = rows_per_dpu;
    }

    // If there are remaining threads create empty partitions
    for (t = split_cnt + 1; t <= nr_of_tasklets; t++) {
        part_info->row_split_tasklet[t] = rows_per_dpu;
    }

}
#endif

/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->row_split);
    free(part_info->row_split_tasklet);
    free(part_info);
}


#endif
