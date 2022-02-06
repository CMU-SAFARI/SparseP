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
    uint32_t *brow_split_tasklet;
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(uint32_t nr_of_dpus, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->brow_split_tasklet = (uint32_t *) calloc(nr_of_dpus * (nr_of_tasklets + 2), sizeof(uint32_t));

    return part_info;
}

#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks across tasklets
 */
void partition_tsklt_by_block(struct DBCSRMatrix *dbcsrMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = 0;
        part_info->brow_split_tasklet[row_offset + 1] = dbcsrMtx->num_block_rows; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = dbcsrMtx->blocks_per_partition[dpu];
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = row_start;
    uint32_t start_row, end_row;
    start_row = 0;

    for (i = 0; i < dbcsrMtx->num_block_rows; i++) {
        uint32_t start_block, end_block;
        start_block = dbcsrMtx->browptr[dpu * (dbcsrMtx->num_block_rows + 1) + i];
        end_block = dbcsrMtx->browptr[dpu * (dbcsrMtx->num_block_rows + 1) + i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += dbcsrMtx->nnz_per_block[acc_blocks + j];
            curr_block++;
        }
        if (curr_block >= block_per_split) {
            ++split_cnt;
            row_start = i + 1;
            if (split_cnt <= nr_of_tasklets) {
                part_info->brow_split_tasklet[row_offset + split_cnt] = row_start;
                curr_block = 0;
                curr_nnz = 0;
            }
        }
    }

    // Fill the last split with remaining elements
    if (curr_block < block_per_split && split_cnt <= nr_of_tasklets) {
        split_cnt++;
        part_info->brow_split_tasklet[row_offset + split_cnt] = dbcsrMtx->num_block_rows; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = dbcsrMtx->num_block_rows; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = dbcsrMtx->num_block_rows; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == dbcsrMtx->num_block_rows && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief balance nnz at block granularity across tasklets
 */
void partition_tsklt_by_nnz(struct DBCSRMatrix *dbcsrMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = 0;
        part_info->brow_split_tasklet[row_offset + 1] = dbcsrMtx->num_block_rows; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = dbcsrMtx->nnzs_per_partition[dpu];
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = 0;
    uint32_t start_row, end_row;
    start_row = 0;
    end_row = dbcsrMtx->num_block_rows;

    for (i = start_row; i < end_row; i++) {
        uint32_t start_block, end_block;
        start_block = dbcsrMtx->browptr[dpu * (dbcsrMtx->num_block_rows + 1) + i];
        end_block = dbcsrMtx->browptr[dpu * (dbcsrMtx->num_block_rows + 1) + i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += dbcsrMtx->nnz_per_block[acc_blocks + j];
            curr_block++;
        }
        if (curr_nnz >= nnz_per_split) {
            ++split_cnt;
            row_start = i + 1;
            if (split_cnt <= nr_of_tasklets) {
                part_info->brow_split_tasklet[row_offset + split_cnt] = row_start;
                curr_block = 0;
                curr_nnz = 0;
            }
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_tasklets) {
        split_cnt++;
        part_info->brow_split_tasklet[row_offset + split_cnt] = dbcsrMtx->num_block_rows; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = dbcsrMtx->num_block_rows; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = dbcsrMtx->num_block_rows; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == dbcsrMtx->num_block_rows && "Invalid partitioning!");

}
#endif


/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->brow_split_tasklet);
    free(part_info);
}



#endif
