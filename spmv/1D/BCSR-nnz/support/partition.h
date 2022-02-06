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
    uint32_t *brow_split;
    uint32_t *blocks_dpu;
    uint32_t *nnzs_dpu;
    uint32_t *brow_split_tasklet;
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(uint32_t nr_of_dpus, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->brow_split = (uint32_t *) malloc((nr_of_dpus + 2) * sizeof(uint32_t));
    part_info->blocks_dpu = (uint32_t *) calloc((nr_of_dpus + 1), sizeof(uint32_t));
    part_info->nnzs_dpu = (uint32_t *) calloc((nr_of_dpus + 1), sizeof(uint32_t));
    part_info->brow_split_tasklet = (uint32_t *) malloc(nr_of_dpus * (nr_of_tasklets + 2) * sizeof(uint32_t));

    return part_info;
}

/** 
 * @brief load-balance nnz across DPUs (block-row granularity)
 */
void partition_by_nnz(struct BCSRMatrix *bcsrMtx, struct partition_info_t * part_info, int nr_of_dpus) {

    if (nr_of_dpus == 1) {
        part_info->brow_split[0] = 0;
        part_info->brow_split[1] = bcsrMtx->num_block_rows;
        part_info->blocks_dpu[0] = bcsrMtx->num_blocks;
        part_info->nnzs_dpu[0] = bcsrMtx->nnz;
        return;
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = bcsrMtx->nnz;
    uint32_t nnz_per_split = nnz_cnt / nr_of_dpus;
    uint32_t curr_nnz = 0;
    uint32_t curr_block = 0;
    uint32_t row_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split[0] = row_start;
    // Iterate at block-row granularity to partition the matrix across DPUs
    for (i = 0; i < bcsrMtx->num_block_rows; i++) {
        for (j = bcsrMtx->browptr[i]; j < bcsrMtx->browptr[i+1]; j++) {
            curr_nnz += bcsrMtx->nnz_per_block[j];
            curr_block++;
        }

        if (curr_nnz >= nnz_per_split) {
            ++split_cnt;
            row_start = i + 1;

            if (split_cnt <= nr_of_dpus) {
                part_info->brow_split[split_cnt] = row_start;
                part_info->blocks_dpu[split_cnt - 1] = curr_block;
                part_info->nnzs_dpu[split_cnt - 1] = curr_nnz;
                curr_block = 0;
                curr_nnz = 0;
            }
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= nr_of_dpus) {
        part_info->brow_split[++split_cnt] = bcsrMtx->num_block_rows;
        part_info->blocks_dpu[split_cnt - 1] = curr_block;
        part_info->nnzs_dpu[split_cnt - 1] = curr_nnz;
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_dpus) {
        part_info->brow_split[nr_of_dpus] = bcsrMtx->num_block_rows;
        part_info->blocks_dpu[nr_of_dpus - 1] += curr_block;
        part_info->nnzs_dpu[nr_of_dpus - 1] += curr_nnz;
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_dpus; i++) {
        part_info->brow_split[i] = bcsrMtx->num_block_rows;
        part_info->blocks_dpu[i - 1] = 0;
        part_info->nnzs_dpu[i - 1] = 0;
    }

    // Sanity Check
    uint32_t total_blocks = 0;
    uint32_t total_nnzs = 0;
    for (i = 0; i < nr_of_dpus; i++) {
        total_blocks += part_info->blocks_dpu[i];
        total_nnzs += part_info->nnzs_dpu[i];
    }
    assert(part_info->brow_split[nr_of_dpus] == bcsrMtx->num_block_rows && "Invalid partitioning!");
    assert(total_blocks == bcsrMtx->num_blocks && "Invalid partitioning!");
    assert(total_nnzs == bcsrMtx->nnz && "Invalid partitioning!");

}



#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks across tasklets (block-row granularity)
 */
void partition_tsklt_by_block(struct BCSRMatrix *bcsrMtx, struct partition_info_t * part_info, int dpu, int nr_of_tasklets, int nr_of_dpus) {


    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    uint32_t block_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = part_info->brow_split[dpu];
        part_info->brow_split_tasklet[row_offset + 1] = part_info->brow_split[dpu+1]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = part_info->blocks_dpu[dpu];
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = part_info->brow_split[dpu];
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = row_start;
    uint32_t start_row, end_row;
    start_row = part_info->brow_split[dpu];
    end_row = part_info->brow_split[dpu+1];

    // Iterate at block-row granularity to partition the matrix across tasklets 
    for (i = start_row; i < end_row; i++) {
        uint32_t start_block, end_block;
        start_block = bcsrMtx->browptr[i];
        end_block = bcsrMtx->browptr[i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += bcsrMtx->nnz_per_block[j];
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
        part_info->brow_split_tasklet[row_offset + split_cnt] = part_info->brow_split[dpu+1]; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = part_info->brow_split[dpu+1]; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = part_info->brow_split[dpu+1]; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == part_info->brow_split[dpu+1] && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief load-balance nnz across tasklets (block-row granularity)
 */
void partition_tsklt_by_nnz(struct BCSRMatrix *bcsrMtx, struct partition_info_t * part_info, int dpu, int nr_of_tasklets, int nr_of_dpus) {

    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    uint32_t block_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = part_info->brow_split[dpu];
        part_info->brow_split_tasklet[row_offset + 1] = part_info->brow_split[dpu+1]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = part_info->nnzs_dpu[dpu];
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = part_info->brow_split[dpu];
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = row_start;
    uint32_t start_row, end_row;
    start_row = part_info->brow_split[dpu];
    end_row = part_info->brow_split[dpu+1];

    // Iterate at block-row granularity to partition the matrix across tasklets 
    for (i = start_row; i < end_row; i++) {
        uint32_t start_block, end_block;
        start_block = bcsrMtx->browptr[i];
        end_block = bcsrMtx->browptr[i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += bcsrMtx->nnz_per_block[j];
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
        part_info->brow_split_tasklet[row_offset + split_cnt] = part_info->brow_split[dpu+1]; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = part_info->brow_split[dpu+1]; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = part_info->brow_split[dpu+1]; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == part_info->brow_split[dpu+1] && "Invalid partitioning!");

}
#endif

/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->brow_split);
    free(part_info->blocks_dpu);
    free(part_info->nnzs_dpu);
    free(part_info->brow_split_tasklet);
    free(part_info);
}



#endif
