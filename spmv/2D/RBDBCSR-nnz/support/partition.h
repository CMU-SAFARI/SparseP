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
#if FG_TRANS
    uint32_t *max_block_rows_per_rank; // max block rows among 64 DPUs per rank
    uint32_t *active_dpus_per_rank; // active DPUs per rank (needed when there are faulty DPUs in the system)
    uint32_t *accum_dpus_ranks; // accumulated active DPUs
#endif
};

/** 
 * @brief allocate data structure for partitioning
 */
struct partition_info_t *partition_init(struct RBDBCSRMatrix *rbdbcsrMtx, uint32_t nr_of_dpus, uint32_t max_ranks, uint32_t nr_of_tasklets) {
    struct partition_info_t *part_info;
    part_info = (struct partition_info_t *) malloc(sizeof(struct partition_info_t));

    part_info->brow_split = (uint32_t *) malloc(rbdbcsrMtx->vert_partitions * (rbdbcsrMtx->horz_partitions + 2) * sizeof(uint32_t));
    part_info->blocks_dpu = (uint32_t *) calloc(rbdbcsrMtx->vert_partitions * (rbdbcsrMtx->horz_partitions + 1), sizeof(uint32_t));
    part_info->nnzs_dpu = (uint32_t *) calloc(rbdbcsrMtx->vert_partitions * (rbdbcsrMtx->horz_partitions + 1), sizeof(uint32_t));

    part_info->brow_split_tasklet = (uint32_t *) malloc(nr_of_dpus * (nr_of_tasklets + 2) * sizeof(uint32_t));

#if FG_TRANS
    part_info->max_block_rows_per_rank = (uint32_t *) calloc(max_ranks, sizeof(uint32_t));
    part_info->active_dpus_per_rank = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
    part_info->accum_dpus_ranks = (uint32_t *) calloc((max_ranks + 1), sizeof(uint32_t));
#endif
    return part_info;
}


/** 
 * @brief load-balance nnzs at block-row granularity across DPUs
 */
void partition_by_nnz(struct RBDBCSRMatrix *rbdbcsrMtx, struct partition_info_t *part_info) {

    uint32_t c, i, j;
    uint32_t total_blocks = 0;
    for(c = 0; c < rbdbcsrMtx->vert_partitions; c++) {
        // Compute the matrix splits.
        uint32_t sptr_offset = c * (rbdbcsrMtx->horz_partitions + 1);
        uint32_t dptr_offset = c * rbdbcsrMtx->horz_partitions;
        uint32_t rptr_offset = c * (rbdbcsrMtx->num_block_rows + 1);
        uint32_t nnz_cnt = rbdbcsrMtx->nnzs_per_vert_partition[c];
        uint32_t nnz_per_split = nnz_cnt / rbdbcsrMtx->horz_partitions;
        uint32_t curr_block = 0;
        uint32_t curr_nnz = 0;
        uint32_t row_start = 0;
        uint32_t split_cnt = 0;

        part_info->brow_split[sptr_offset] = row_start;
        for (i = 0; i < rbdbcsrMtx->num_block_rows; i++) {
            curr_block += rbdbcsrMtx->browptr[rptr_offset + i+1] - rbdbcsrMtx->browptr[rptr_offset + i];
            for (j = rbdbcsrMtx->browptr[rptr_offset + i]; j < rbdbcsrMtx->browptr[rptr_offset + i+1]; j++)
                curr_nnz += rbdbcsrMtx->nnz_per_block[total_blocks+j];

            if (curr_nnz >= nnz_per_split) {
                row_start = i + 1;
                ++split_cnt;
                if (split_cnt <= rbdbcsrMtx->horz_partitions) {
                    part_info->brow_split[sptr_offset + split_cnt] = row_start;
                    part_info->blocks_dpu[dptr_offset + split_cnt - 1] = curr_block;
                    part_info->nnzs_dpu[dptr_offset + split_cnt - 1] = curr_nnz;
                    curr_block = 0;
                    curr_nnz = 0;
                }
                //curr_nnz = 0;
            }
        }

        // Fill the last split with remaining elements
        if (curr_nnz < nnz_per_split && split_cnt <= rbdbcsrMtx->horz_partitions) {
            split_cnt++;
            part_info->brow_split[sptr_offset + split_cnt] = rbdbcsrMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + split_cnt - 1] = curr_block;
            part_info->nnzs_dpu[dptr_offset + split_cnt - 1] = curr_nnz;
        }

        // If there are any remaining rows merge them in last partition
        if (split_cnt > rbdbcsrMtx->horz_partitions) {
            part_info->brow_split[sptr_offset + rbdbcsrMtx->horz_partitions] = rbdbcsrMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + rbdbcsrMtx->horz_partitions - 1] += curr_block;
            part_info->nnzs_dpu[dptr_offset + rbdbcsrMtx->horz_partitions - 1] += curr_nnz;
        }

        // If there are remaining threads create empty partitions
        for (i = split_cnt + 1; i <= rbdbcsrMtx->horz_partitions; i++) {
            part_info->brow_split[sptr_offset + i] = rbdbcsrMtx->num_block_rows;
            part_info->blocks_dpu[dptr_offset + i - 1] = 0;
            part_info->nnzs_dpu[dptr_offset + i - 1] = 0;
        }

        total_blocks += rbdbcsrMtx->blocks_per_vert_partition[c];
    }


    // Sanity Check
    total_blocks = 0;
    uint32_t total_nnzs = 0;
    for (c = 0; c < rbdbcsrMtx->vert_partitions; c++) {
        for (i = 0; i < rbdbcsrMtx->horz_partitions; i++) {
            total_blocks += part_info->blocks_dpu[c * rbdbcsrMtx->horz_partitions + i];
            total_nnzs += part_info->nnzs_dpu[c * rbdbcsrMtx->horz_partitions + i];
        }
    }
    assert(total_blocks == rbdbcsrMtx->nblocks && "Invalid partitioning!");
    assert(total_nnzs == rbdbcsrMtx->nnz && "Invalid partitioning!");

}


#if BLNC_TSKLT_BLOCK
/** 
 * @brief load-balance blocks at block-row granularity across tasklets
 */
void partition_tsklt_by_block(struct RBDBCSRMatrix *rbdbcsrMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks, int prev_block_rows_dpu, int block_rows_dpu, int dpu_col_indx) {

    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    uint32_t block_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = prev_block_rows_dpu;
        part_info->brow_split_tasklet[row_offset + 1] = prev_block_rows_dpu + block_rows_dpu; 
        goto SANITY_CHECK; 
    }

    // Compute the matrix splits.
    uint32_t block_cnt = part_info->blocks_dpu[dpu];
    uint32_t block_per_split = block_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = prev_block_rows_dpu;
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = row_start;
    uint32_t start_row, end_row;
    start_row = 0;

    for (i = prev_block_rows_dpu; i < prev_block_rows_dpu + block_rows_dpu; i++) {
        uint32_t start_block, end_block;
        start_block = rbdbcsrMtx->browptr[dpu_col_indx * (rbdbcsrMtx->num_block_rows + 1) + i];
        end_block = rbdbcsrMtx->browptr[dpu_col_indx * (rbdbcsrMtx->num_block_rows + 1) + i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += rbdbcsrMtx->nnz_per_block[acc_blocks + j];
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
        part_info->brow_split_tasklet[row_offset + split_cnt] = prev_block_rows_dpu + block_rows_dpu; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = prev_block_rows_dpu + block_rows_dpu; 
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = prev_block_rows_dpu + block_rows_dpu; 
    }

    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == (prev_block_rows_dpu + block_rows_dpu)  && "Invalid partitioning!");

}
#endif

#if BLNC_TSKLT_NNZ
/** 
 * @brief load-balance nnzs at block-row granularity across tasklets
 */ 
void partition_tsklt_by_nnz(struct RBDBCSRMatrix *rbdbcsrMtx, struct partition_info_t *part_info, int dpu, int nr_of_tasklets, int nr_of_dpus, int acc_blocks, int prev_block_rows_dpu, int block_rows_dpu, int dpu_col_indx) {

    uint32_t row_offset = dpu * (nr_of_tasklets + 2);
    //uint32_t block_offset = dpu * (nr_of_tasklets + 1);
    if (nr_of_tasklets == 1) {
        part_info->brow_split_tasklet[row_offset + 0] = prev_block_rows_dpu;
        part_info->brow_split_tasklet[row_offset + 1] = prev_block_rows_dpu + block_rows_dpu; 
        goto SANITY_CHECK; 
    }


    // Compute the matrix splits.
    uint32_t nnz_cnt = part_info->nnzs_dpu[dpu];
    uint32_t nnz_per_split = nnz_cnt / nr_of_tasklets;
    uint32_t curr_block = 0;
    uint32_t curr_nnz = 0;
    uint32_t row_start = prev_block_rows_dpu;
    uint32_t split_cnt = 0;
    uint32_t i, j;

    part_info->brow_split_tasklet[row_offset + 0] = row_start;
    uint32_t start_row, end_row;
    start_row = prev_block_rows_dpu;
    end_row = prev_block_rows_dpu + block_rows_dpu;

    for (i = start_row; i < end_row; i++) {
        uint32_t start_block, end_block;
        start_block = rbdbcsrMtx->browptr[dpu_col_indx * (rbdbcsrMtx->num_block_rows + 1) + i];
        end_block = rbdbcsrMtx->browptr[dpu_col_indx * (rbdbcsrMtx->num_block_rows + 1) + i+1];
        for (j = start_block; j < end_block; j++) {
            curr_nnz += rbdbcsrMtx->nnz_per_block[acc_blocks + j];
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
        part_info->brow_split_tasklet[row_offset + split_cnt] = prev_block_rows_dpu + block_rows_dpu; 
    }

    // If there are any remaining blocks merge them in last partition
    if (split_cnt > nr_of_tasklets) {
        part_info->brow_split_tasklet[row_offset + nr_of_tasklets] = prev_block_rows_dpu + block_rows_dpu;
    }

    // If there are remaining threads create empty partitions
    for (i = split_cnt + 1; i <= nr_of_tasklets; i++) {
        part_info->brow_split_tasklet[row_offset + i] = prev_block_rows_dpu + block_rows_dpu;
    }


    // Sanity Check
SANITY_CHECK:    
    printf("");
    assert(part_info->brow_split_tasklet[row_offset + nr_of_tasklets] == (prev_block_rows_dpu + block_rows_dpu)  && "Invalid partitioning!");

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
#if FG_TRANS
    free(part_info->max_block_rows_per_rank); 
    free(part_info->active_dpus_per_rank); 
    free(part_info->accum_dpus_ranks);
#endif
    free(part_info);
}


#endif
