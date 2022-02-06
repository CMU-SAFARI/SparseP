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
    uint32_t *block_split_tasklet;
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(uint32_t nr_of_dpus, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->block_split_tasklet = (uint32_t *) malloc(nr_of_dpus * (nr_of_tasklets + 2) * sizeof(uint32_t));

    return part_info;
}

#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks across tasklets
 */
void partition_tsklt_by_block(struct DBCOOMatrix *dbcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = 0;
        part_info->block_split_tasklet[block_offset + 1] = dbcooMtx->blocks_per_partition[dpu]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = dbcooMtx->blocks_per_partition[dpu]; 
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t rest_blocks = block_cnt % nr_of_tasklets;
    uint32_t blocks_per_tasklet, nnz_per_tasklet; 
    uint32_t i,j;

    part_info->block_split_tasklet[block_offset + 0] = 0;
    for(i = 0; i < nr_of_tasklets; i++) {
        blocks_per_tasklet = block_per_split; 
        if (i < rest_blocks)
            blocks_per_tasklet++;
        part_info->block_split_tasklet[block_offset + i+1] = part_info->block_split_tasklet[block_offset + i] + blocks_per_tasklet;
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");

    uint32_t total_blocks = 0;
    for (i = 0; i < nr_of_tasklets; i++) {
        total_blocks += (part_info->block_split_tasklet[block_offset + i+1] - part_info->block_split_tasklet[block_offset + i]);
    }
    assert(total_blocks == dbcooMtx->blocks_per_partition[dpu] && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief load-balance nnz across tasklets
 */
void partition_tsklt_by_nnz(struct DBCOOMatrix *dbcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = 0;
        part_info->block_split_tasklet[block_offset + 1] = dbcooMtx->blocks_per_partition[dpu]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = dbcooMtx->nnzs_per_partition[dpu];
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_nnz = 0;
    uint32_t block_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i;

    part_info->block_split_tasklet[block_offset + 0] = block_start;
    for (i = 0; i < dbcooMtx->blocks_per_partition[dpu]; i++) {
        curr_nnz += dbcooMtx->nnz_per_block[acc_blocks + i];
        if (curr_nnz >= nnz_per_split) {
            ++split_cnt;
            block_start = i + 1;
            if (split_cnt <= nr_of_tasklets) {
                part_info->block_split_tasklet[block_offset + split_cnt] = block_start;
                curr_nnz = 0;
            }
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_tasklets) {
        split_cnt++;
        part_info->block_split_tasklet[block_offset + split_cnt] = dbcooMtx->blocks_per_partition[dpu]; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->block_split_tasklet[block_offset + nr_of_tasklets] = dbcooMtx->blocks_per_partition[dpu];
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->block_split_tasklet[block_offset + i] = dbcooMtx->blocks_per_partition[dpu];
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");

    uint32_t total_blocks = 0;
    for (i = 0; i < nr_of_tasklets; i++) {
        total_blocks += (part_info->block_split_tasklet[block_offset + i+1] - part_info->block_split_tasklet[block_offset + i]);
    }
    assert(total_blocks == dbcooMtx->blocks_per_partition[dpu] && "Invalid partitioning!");

}
#endif



/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->block_split_tasklet);
    free(part_info);
}



#endif
