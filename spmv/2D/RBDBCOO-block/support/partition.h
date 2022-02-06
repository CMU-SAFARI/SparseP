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
    uint32_t *block_split_tasklet;
#if FG_TRANS
    uint32_t *max_block_rows_per_rank; // max block rows among 64 DPUs per rank
    uint32_t *active_dpus_per_rank; // active DPUs per rank (needed when there are faulty DPUs in the system)
    uint32_t *accum_dpus_ranks; // accumulated active DPUs
#endif
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(struct RBDBCOOMatrix *rbdbcooMtx, uint32_t nr_of_dpus, uint32_t max_ranks, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->brow_split = (uint32_t *) malloc(2 * rbdbcooMtx->vert_partitions * (rbdbcooMtx->horz_partitions + 2) * sizeof(uint32_t));
    part_info->blocks_dpu = (uint32_t *) calloc(rbdbcooMtx->vert_partitions * (rbdbcooMtx->horz_partitions + 1), sizeof(uint32_t));
    part_info->nnzs_dpu = (uint32_t *) calloc(rbdbcooMtx->vert_partitions * (rbdbcooMtx->horz_partitions + 1), sizeof(uint32_t));

    part_info->block_split_tasklet = (uint32_t *) malloc(nr_of_dpus * (nr_of_tasklets + 2) * sizeof(uint32_t));

#if FG_TRANS
    part_info->max_block_rows_per_rank = (uint32_t *) calloc(max_ranks, sizeof(uint32_t));
    part_info->active_dpus_per_rank = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
    part_info->accum_dpus_ranks = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
#endif
    return part_info;
}

/** 
 * @brief load-balance blocks across DPUs
 */
void partition_by_block(struct RBDBCOOMatrix *rbdbcooMtx, struct partition_info_t *part_info) {

    uint32_t c, i, j;
    uint32_t total_blocks = 0;
    uint32_t total_block_rows = 0;
    for(c = 0; c < rbdbcooMtx->vert_partitions; c++) {
        // Compute the matrix splits.
        uint32_t sptr_offset = c * (rbdbcooMtx->horz_partitions + 1);
        uint32_t dptr_offset = c * rbdbcooMtx->horz_partitions;
        uint32_t rptr_offset = 2 * c * rbdbcooMtx->horz_partitions;
        uint32_t block_cnt = rbdbcooMtx->blocks_per_vert_partition[c];
        uint32_t block_per_split = block_cnt / rbdbcooMtx->horz_partitions;
        uint32_t curr_block = 0;
        uint32_t curr_nnz = 0;
        uint32_t row_start = 0;
        uint32_t split_cnt = 0;

        part_info->brow_split[rptr_offset] = row_start;
        for (i = 0; i < rbdbcooMtx->blocks_per_vert_partition[c]; i++) {
            curr_block++;
            curr_nnz += rbdbcooMtx->nnz_per_block[total_blocks+i];
            if (curr_block >= block_per_split) {
                row_start = rbdbcooMtx->bind[total_blocks + i].rowind + 1;
                ++split_cnt;
                if (split_cnt <= rbdbcooMtx->horz_partitions) {
                    part_info->brow_split[rptr_offset + 2 * (split_cnt - 1) + 1] = row_start;
                    if (i == (rbdbcooMtx->blocks_per_vert_partition[c] - 1)) { 
                        part_info->brow_split[rptr_offset + 2 * split_cnt] = row_start;
                    } else if (rbdbcooMtx->bind[total_blocks + i].rowind == rbdbcooMtx->bind[total_blocks + i + 1].rowind) {
                        part_info->brow_split[rptr_offset + 2 * split_cnt] = row_start - 1;
                    } else {
                        part_info->brow_split[rptr_offset + 2 * split_cnt] = row_start;
                    }
                    part_info->blocks_dpu[dptr_offset + split_cnt - 1] = curr_block;
                    part_info->nnzs_dpu[dptr_offset + split_cnt - 1] = curr_nnz;
                    curr_block = 0;
                    curr_nnz = 0;
                }
                //curr_block = 0;
                //curr_nnz = 0;
            }
        }


        // Fill the last split with remaining elements
        if (curr_block < block_per_split && split_cnt <= rbdbcooMtx->horz_partitions) {
            split_cnt++;
            part_info->brow_split[rptr_offset + 2 * (split_cnt - 1) + 1] = rbdbcooMtx->num_block_rows;
            part_info->brow_split[rptr_offset + 2 * split_cnt] = rbdbcooMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + split_cnt - 1] = curr_block;
            part_info->nnzs_dpu[dptr_offset + split_cnt - 1] = curr_nnz;
        }

        // If there are any remaining rows merge them in last partition
        if (split_cnt > rbdbcooMtx->horz_partitions) {
            part_info->brow_split[rptr_offset + 2 * (rbdbcooMtx->horz_partitions - 1) + 1] = rbdbcooMtx->num_block_rows;
            part_info->brow_split[rptr_offset + 2 * rbdbcooMtx->horz_partitions] = rbdbcooMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + rbdbcooMtx->horz_partitions - 1] += curr_block;
            part_info->nnzs_dpu[dptr_offset + rbdbcooMtx->horz_partitions - 1] += curr_nnz;
        }

        // If there are remaining threads create empty partitions
        for (i = split_cnt + 1; i <= rbdbcooMtx->horz_partitions; i++) {
            part_info->brow_split[rptr_offset + 2 * (i - 1) + 1] = rbdbcooMtx->num_block_rows;
            part_info->brow_split[rptr_offset + 2 * i] = rbdbcooMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + i - 1] = 0;
            part_info->nnzs_dpu[dptr_offset + i - 1] = 0;
        }

        total_blocks += rbdbcooMtx->blocks_per_vert_partition[c];
        total_block_rows += rbdbcooMtx->num_block_rows;
    }


    // Sanity Check
    total_blocks = 0;
    uint32_t total_nnzs = 0;
    for (c = 0; c < rbdbcooMtx->vert_partitions; c++) {
        for (i = 0; i < rbdbcooMtx->horz_partitions; i++) {
            total_blocks += part_info->blocks_dpu[c * rbdbcooMtx->horz_partitions + i];
            total_nnzs += part_info->nnzs_dpu[c * rbdbcooMtx->horz_partitions + i];
        }
    }
    assert(total_blocks == rbdbcooMtx->nblocks && "Invalid partitioning!");
    assert(total_nnzs == rbdbcooMtx->nnz && "Invalid partitioning!");

}



#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks across tasklets
 */ 
void partition_tsklt_by_block(struct RBDBCOOMatrix *rbdbcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = 0;
        part_info->block_split_tasklet[block_offset + 1] = part_info->blocks_dpu[dpu]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = part_info->blocks_dpu[dpu]; 
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t rest_blocks = block_cnt % nr_of_tasklets;
    uint32_t blocks_per_tasklet, nnz_per_tasklet; 
    uint32_t i;

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
    assert(total_blocks == part_info->blocks_dpu[dpu] && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief load-balance nnzs across tasklets
 */ 
void partition_tsklt_by_nnz(struct RBDBCOOMatrix *rbdbcooMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks) {

    uint32_t block_offset = dpu * (nr_of_tasklets + 2);
    //uint32_t nnz_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->block_split_tasklet[block_offset + 0] = 0;
        part_info->block_split_tasklet[block_offset + 1] = part_info->blocks_dpu[dpu]; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t nnz_cnt = part_info->nnzs_dpu[dpu];
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_nnz = 0;
    uint32_t block_start = 0;
    uint32_t split_cnt = 0;
    uint32_t i;

    part_info->block_split_tasklet[block_offset + 0] = block_start;
    for (i = 0; i < part_info->blocks_dpu[dpu]; i++) {
        curr_nnz += rbdbcooMtx->nnz_per_block[acc_blocks + i];
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
        part_info->block_split_tasklet[block_offset + split_cnt] = part_info->blocks_dpu[dpu]; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->block_split_tasklet[block_offset + nr_of_tasklets] = part_info->blocks_dpu[dpu];
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->block_split_tasklet[block_offset + i] = part_info->blocks_dpu[dpu]; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    uint32_t total_blocks = 0;
    for (i = 0; i < nr_of_tasklets; i++) {
        total_blocks += (part_info->block_split_tasklet[block_offset + i+1] - part_info->block_split_tasklet[block_offset + i]);
    }
    assert(total_blocks == part_info->blocks_dpu[dpu] && "Invalid partitioning!");

}
#endif

/*
 * @brief deallocate partition_info data
 */
void partition_free(struct partition_info_t *part_info) {
    free(part_info->brow_split);
    free(part_info->blocks_dpu);
    free(part_info->nnzs_dpu);
    free(part_info->block_split_tasklet);
#if FG_TRANS
    free(part_info->max_block_rows_per_rank); 
    free(part_info->active_dpus_per_rank); 
    free(part_info->accum_dpus_ranks);
#endif
    free(part_info);
}


#endif
