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
    uint32_t *block_split;
    uint32_t *nnzs_dpu;
    uint32_t *block_split_tasklet;
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(uint32_t nr_of_dpus, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->block_split = (uint32_t *) malloc((nr_of_dpus + 2) * sizeof(uint32_t));
    part_info->nnzs_dpu = (uint32_t *) calloc((nr_of_dpus + 1), sizeof(uint32_t));
    part_info->block_split_tasklet = (uint32_t *) malloc(nr_of_dpus * (nr_of_tasklets + 2) * sizeof(uint32_t));

    return part_info;
}

/** 
 * @brief load-balance blocks across DPUs
 */
void partition_by_blocks(struct BCOOMatrix *bcooMtx, struct partition_info_t * part_info, int nr_of_dpus) {

    if (nr_of_dpus == 1) {
        part_info->block_split[0] = 0;
        part_info->block_split[1] = bcooMtx->num_blocks;
        part_info->nnzs_dpu[0] = bcooMtx->nnz;
        return;
    }

    // Compute the matrix splits.
    uint32_t block_cnt = bcooMtx->num_blocks;
    uint32_t block_per_split = block_cnt / nr_of_dpus;
    uint32_t rest_blocks = block_cnt % nr_of_dpus;
    uint32_t blocks_per_dpu; 
    uint32_t i,j;

    part_info->block_split[0] = 0;
    for(i = 0; i < nr_of_dpus; i++) {
        blocks_per_dpu = block_per_split; 
        if (i < rest_blocks)
            blocks_per_dpu++;
        part_info->block_split[i+1] = part_info->block_split[i] + blocks_per_dpu;
        for(j = part_info->block_split[i]; j < part_info->block_split[i+1]; j++) {
            part_info->nnzs_dpu[i] += bcooMtx->nnz_per_block[j]; 
        }
    }

    // Sanity Check
    uint32_t total_blocks = 0;
    uint32_t total_nnzs = 0;
    for (i = 0; i < nr_of_dpus; i++) {
        total_blocks += (part_info->block_split[i+1] - part_info->block_split[i]);
        total_nnzs += part_info->nnzs_dpu[i];
    }
    assert(total_blocks == bcooMtx->num_blocks && "Invalid partitioning!");
    assert(total_nnzs == bcooMtx->nnz && "Invalid partitioning!");

}

#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks across tasklets
 */
void partition_tsklt_by_block(struct BCOOMatrix *bcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = part_info->block_split[dpu];
        part_info->block_split_tasklet[block_offset + 1] = part_info->block_split[dpu+1]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = (part_info->block_split[dpu+1] - part_info->block_split[dpu]); 
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t rest_blocks = block_cnt % nr_of_tasklets;
    uint32_t blocks_per_tasklet, nnz_per_tasklet; 
    uint32_t i,j;

    part_info->block_split_tasklet[block_offset + 0] = part_info->block_split[dpu];
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
    assert(total_blocks == (part_info->block_split[dpu+1]-part_info->block_split[dpu]) && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief load-balance nnz at block granularity across tasklets
 */
void partition_tsklt_by_nnz(struct BCOOMatrix *bcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = part_info->block_split[dpu];
        part_info->block_split_tasklet[block_offset + 1] = part_info->block_split[dpu+1]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = part_info->nnzs_dpu[dpu]; 
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_nnz = 0;
    uint32_t block_start = part_info->block_split[dpu];
    uint32_t split_cnt = 0;
    uint32_t i;

    part_info->block_split_tasklet[block_offset + 0] = block_start;
    for (i = part_info->block_split[dpu]; i < part_info->block_split[dpu+1]; i++) {
        curr_nnz += bcooMtx->nnz_per_block[i];
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
        part_info->block_split_tasklet[block_offset + split_cnt] = part_info->block_split[dpu+1]; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->block_split_tasklet[block_offset + nr_of_tasklets] = part_info->block_split[dpu+1]; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->block_split_tasklet[block_offset + i] = part_info->block_split[dpu+1];
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    uint32_t total_blocks = 0;
    for (i = 0; i < nr_of_tasklets; i++) {
        total_blocks += (part_info->block_split_tasklet[block_offset + i+1] - part_info->block_split_tasklet[block_offset + i]);
    }
    assert(total_blocks == (part_info->block_split[dpu+1]-part_info->block_split[dpu]) && "Invalid partitioning!");

}
#endif

/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->block_split);
    free(part_info->nnzs_dpu);
    free(part_info->block_split_tasklet);
    free(part_info);
}



#endif
