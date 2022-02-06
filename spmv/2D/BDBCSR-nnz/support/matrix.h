/**
 * Christina Giannoula
 * cgiannoula: christina.giann@gmail.com
 */

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "utils.h"

/**
 * @brief COO matrix format 
 */
struct COOMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t *rows;
    struct elem_t *nnzs;
};

/**
 * @brief BDCSR matrix format 
 * 2D-partitioned matrix with variable-sized tiles and CSR on each vertical tile
 */
struct BDCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t max_tile_width;
    uint32_t *nnzs_per_vert_partition;
    uint32_t *nnzs_per_column;
    uint32_t *vert_tile_widths;
    uint32_t *drowptr;
    uint32_t *dcolind;
    val_dt *dval;
};


/**
 * @brief BDBCSR matrix format 
 * 2D-partitioned matrix with variable-sized tiles and BCSR on each vertical tile
 */
struct BDBCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t nblocks;
    uint32_t max_tile_width;
    uint32_t *nnzs_per_vert_partition;
    uint32_t *nnzs_per_column;
    uint32_t *blocks_per_vert_partition;
    uint32_t *vert_tile_widths;
    uint32_t *nnz_per_block;  // nnz per block
    uint64_t row_block_size;
    uint64_t col_block_size;
    uint32_t num_block_rows;
    uint32_t num_block_cols;
    uint32_t num_rows_left;
    uint32_t* browptr;  // row pointer
    uint32_t* bcolind;  // column indices
    val_dt* bval;       // nonzeros
};


/**
 * @brief read matrix from input fileName in COO format 
 * @param filename to read matrix (mtx format)
 */
struct COOMatrix *readCOOMatrix(const char* fileName) {
    struct COOMatrix *cooMtx;
    cooMtx = (struct COOMatrix *) malloc(sizeof(struct COOMatrix));
    FILE* fp = fopen(fileName, "r");
    uint32_t rowindx, colindx;
    int32_t val;
    char *line; 
    char *token; 
    line = (char *) malloc(1000 * sizeof(char));
    int done = false;
    int i = 0;

    while(fgets(line, 1000, fp) != NULL){
        token = strtok(line, " ");

        if(token[0] == '%'){
            ;
        } else if (done == false) {
            cooMtx->nrows = atoi(token);
            token = strtok(NULL, " ");
            cooMtx->ncols = atoi(token);
            token = strtok(NULL, " ");
            cooMtx->nnz = atoi(token);
            printf("[INFO] %s: %u Rows, %u Cols, %u NNZs\n", strrchr(fileName, '/')+1, cooMtx->nrows, cooMtx->ncols, cooMtx->nnz);
            if((cooMtx->nrows % (8 / byte_dt)) != 0) { // Padding needed
                cooMtx->nrows += ((8 / byte_dt) - (cooMtx->nrows % (8 / byte_dt)));
            }
            if((cooMtx->ncols % (8 / byte_dt)) != 0) { // Padding needed
                cooMtx->ncols += ((8 / byte_dt) - (cooMtx->ncols % (8 / byte_dt)));
            }

            cooMtx->rows = (uint32_t *) calloc((cooMtx->nrows), sizeof(uint32_t));
            cooMtx->nnzs = (struct elem_t *) calloc((cooMtx->nnz+8), sizeof(struct elem_t));
            done = true;
        } else {
            rowindx = atoi(token);
            token = strtok(NULL, " ");
            colindx = atoi(token);
            token = strtok(NULL, " ");
            val = (val_dt) (rand()%4 + 1);

            cooMtx->nnzs[i].rowind = rowindx - 1; // Convert indexes to start at 0
            cooMtx->nnzs[i].colind = colindx - 1; // Convert indexes to start at 0
            cooMtx->nnzs[i].val = val; 
            i++;
        }
    }

    free(line);
    fclose(fp);
    return cooMtx;
}

/** 
 * brief Comparator for Quicksort
 */
int comparator(void *a, void *b) {
    if (((struct elem_t *)a)->rowind < ((struct elem_t *)b)->rowind) {
        return -1;
    } else if (((struct elem_t *)a)->rowind > ((struct elem_t *)b)->rowind) {
        return 1;
    } else {
        return (((struct elem_t *)a)->colind - ((struct elem_t *)b)->colind);
    }
}

/**
 * @brief Sort CooMatrix
 * @param matrix in COO format
 */
void sortCOOMatrix(struct COOMatrix *cooMtx) {

    qsort(cooMtx->nnzs, cooMtx->nnz, sizeof(struct elem_t), comparator);

    int prev_row = cooMtx->nnzs[0].rowind;
    int cur_nnz = 0;
    for(unsigned int n = 0; n < cooMtx->nnz; n++) {
        if(cooMtx->nnzs[n].rowind == prev_row)
            cur_nnz++;
        else {
            cooMtx->rows[prev_row] = cur_nnz;
            prev_row = cooMtx->nnzs[n].rowind;
            cur_nnz = 1;
        }
    }
    cooMtx->rows[prev_row] = cur_nnz;

    cur_nnz = 0;
    for(unsigned int r = 0; r < cooMtx->nrows; r++) {
        cur_nnz += cooMtx->rows[r];
    }
    assert(cur_nnz == cooMtx->nnz);
}

/**
 * @brief deallocate matrix in COO format 
 * @param matrix in COO format
 */
void freeCOOMatrix(struct COOMatrix *cooMtx) {
    free(cooMtx->rows);
    free(cooMtx->nnzs);
    free(cooMtx);
}

/**
 * @brief convert matrix from COO to BDCSR format 
 * @param matrix in COO format
 * @param horz_partitions, vert_partitions
 */
struct BDCSRMatrix *coo2bdcsr(struct COOMatrix *cooMtx, int horz_partitions, int vert_partitions) {

    struct BDCSRMatrix *bdcsrMtx;
    bdcsrMtx = (struct BDCSRMatrix *) malloc(sizeof(struct BDCSRMatrix));

    bdcsrMtx->nrows = cooMtx->nrows;
    bdcsrMtx->ncols = cooMtx->ncols;
    bdcsrMtx->nnz = cooMtx->nnz;

    bdcsrMtx->npartitions = vert_partitions;
    bdcsrMtx->horz_partitions = horz_partitions;
    bdcsrMtx->vert_partitions = vert_partitions;
    bdcsrMtx->max_tile_width = 0;

    // Find nnzs per column
    bdcsrMtx->vert_tile_widths = (uint32_t *) calloc((vert_partitions + 1), sizeof(uint32_t));
    bdcsrMtx->nnzs_per_vert_partition = (uint32_t *) calloc(vert_partitions, sizeof(uint32_t));
    bdcsrMtx->nnzs_per_column = (uint32_t *) calloc((bdcsrMtx->ncols), sizeof(uint32_t));
    for (uint32_t n = 0; n < cooMtx->nnz; n++) {
        bdcsrMtx->nnzs_per_column[cooMtx->nnzs[n].colind]++;        
    }

    uint32_t nnz_per_split = bdcsrMtx->nnz / vert_partitions; 
    uint32_t curr_nnz = 0;
    uint32_t column_start = 0;
    uint32_t split_cnt = 0;
    uint32_t t;

    bdcsrMtx->vert_tile_widths[0] = column_start;
    for (t = 0; t < bdcsrMtx->ncols; t++) {
        curr_nnz += bdcsrMtx->nnzs_per_column[t];
        if (curr_nnz >= nnz_per_split) {
            column_start = t + 1;
            ++split_cnt;
            if (split_cnt <= bdcsrMtx->vert_partitions) {
                bdcsrMtx->vert_tile_widths[split_cnt] = column_start;
                bdcsrMtx->nnzs_per_vert_partition[split_cnt-1] = curr_nnz;
            }
            curr_nnz = 0;
        }
    }

    // Fill the last split with remaining elements
    if (curr_nnz < nnz_per_split && split_cnt <= bdcsrMtx->vert_partitions) {
        bdcsrMtx->vert_tile_widths[++split_cnt] = bdcsrMtx->ncols;
        bdcsrMtx->nnzs_per_vert_partition[split_cnt-1] = curr_nnz;
    }

    // If there are any remaining rows merge them in last partition
    if (split_cnt > bdcsrMtx->vert_partitions) {
        bdcsrMtx->vert_tile_widths[bdcsrMtx->vert_partitions] = bdcsrMtx->ncols;
        bdcsrMtx->nnzs_per_vert_partition[bdcsrMtx->vert_partitions-1] += bdcsrMtx->nnzs_per_vert_partition[split_cnt-1]; 
    }

    // If there are remaining threads create empty partitions
    for (t = split_cnt + 1; t <= bdcsrMtx->vert_partitions; t++) {
        bdcsrMtx->vert_tile_widths[t] = bdcsrMtx->ncols;
        bdcsrMtx->nnzs_per_vert_partition[t-1] = 0;
    }

    // Find maximum width of vertical partitions 
    uint64_t total_nnzs = 0;
    bdcsrMtx->max_tile_width = 0;
    for (t = 0; t < bdcsrMtx->vert_partitions; t++) {
        total_nnzs += bdcsrMtx->nnzs_per_vert_partition[t]; 
        if((bdcsrMtx->vert_tile_widths[t+1] - bdcsrMtx->vert_tile_widths[t]) > bdcsrMtx->max_tile_width)
            bdcsrMtx->max_tile_width = bdcsrMtx->vert_tile_widths[t+1] - bdcsrMtx->vert_tile_widths[t];
    }
    assert(total_nnzs == bdcsrMtx->nnz && "Wrong partitioning");
    assert(bdcsrMtx->vert_tile_widths[vert_partitions] == bdcsrMtx->ncols && "Wrong partitioning");


    // Count nnzs per partition
    uint32_t p_row, p_col, local_col;
    bdcsrMtx->drowptr = (uint32_t *) calloc((bdcsrMtx->nrows + 2) * bdcsrMtx->npartitions, sizeof(uint32_t));
    bdcsrMtx->dcolind = (uint32_t *) malloc((bdcsrMtx->nnz + 2) * sizeof(uint32_t));
    bdcsrMtx->dval = (val_dt *) calloc((bdcsrMtx->nnz + 8),  sizeof(val_dt));

    for (uint32_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind; 
        p_col = 0;
        while(bdcsrMtx->vert_tile_widths[p_col+1] <= cooMtx->nnzs[n].colind)
            p_col++;
        bdcsrMtx->drowptr[p_col * (bdcsrMtx->nrows + 1) + p_row]++;
    }

    // Normalize local rowptrs within each partition 
    for (uint32_t p = 0; p < bdcsrMtx->npartitions; p++) {
        uint32_t sumBeforeNextRow = 0;
        for(unsigned int rowIndx = 0; rowIndx < bdcsrMtx->nrows; ++rowIndx) {
            uint32_t sumBeforeRow = sumBeforeNextRow;
            sumBeforeNextRow += bdcsrMtx->drowptr[p * (bdcsrMtx->nrows + 1) + rowIndx];
            bdcsrMtx->drowptr[p * (bdcsrMtx->nrows + 1) + rowIndx] = sumBeforeRow;
        }
        bdcsrMtx->drowptr[p * (bdcsrMtx->nrows + 1) + bdcsrMtx->nrows] = sumBeforeNextRow;
        assert(bdcsrMtx->drowptr[p * (bdcsrMtx->nrows + 1) + bdcsrMtx->nrows] == bdcsrMtx->nnzs_per_vert_partition[p]);
    }

    // Fill nnzs to partitions
    uint64_t *global_nnzs = (uint64_t *) calloc(bdcsrMtx->npartitions, sizeof(uint64_t));
    uint64_t *local_nnzs = (uint64_t *) calloc(bdcsrMtx->npartitions, sizeof(uint64_t));
    total_nnzs = 0;
    for (uint32_t p = 0; p < bdcsrMtx->npartitions; p++) {
        global_nnzs[p] = total_nnzs;
        total_nnzs += bdcsrMtx->nnzs_per_vert_partition[p];
    }

    for (uint64_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind; 
        p_col = 0;
        while(bdcsrMtx->vert_tile_widths[p_col+1] <= cooMtx->nnzs[n].colind)
            p_col++;
        local_col = cooMtx->nnzs[n].colind - (bdcsrMtx->vert_tile_widths[p_col]); 
        bdcsrMtx->dcolind[global_nnzs[p_col] + local_nnzs[p_col]] = local_col; 
        bdcsrMtx->dval[global_nnzs[p_col] + local_nnzs[p_col]] = cooMtx->nnzs[n].val;
        local_nnzs[p_col]++;
    }

    free(global_nnzs);
    free(local_nnzs);

    return bdcsrMtx;
}

/**
 * @brief deallocate matrix in BDCSR format 
 * @param matrix in BDCSR format
 */ 
void freeBDCSRMatrix(struct BDCSRMatrix *bdcsrMtx) {
    free(bdcsrMtx->nnzs_per_column);
    free(bdcsrMtx->nnzs_per_vert_partition);
    free(bdcsrMtx->vert_tile_widths);
    free(bdcsrMtx->drowptr);
    free(bdcsrMtx->dcolind);
    free(bdcsrMtx->dval);
    free(bdcsrMtx);
}



/**
 * @brief convert matrix from BDCSR to BDBCSR format 
 * @param matrix in BDCSR format
 * @param horz_partitions, vert_partitions, row_block_size, col_block_size
 */
struct BDBCSRMatrix *bdcsr2bdbcsr(struct BDCSRMatrix *bdcsrMtx, uint32_t row_block_size, uint32_t col_block_size) {
    struct BDBCSRMatrix *bdbcsrMtx;
    bdbcsrMtx = (struct BDBCSRMatrix *) malloc(sizeof(struct BDBCSRMatrix));

    uint32_t num_block_rows = (bdcsrMtx->nrows + row_block_size - 1) / row_block_size;
    uint32_t num_block_cols = (bdcsrMtx->max_tile_width + col_block_size - 1) / col_block_size; 
    uint32_t num_rows_left = bdcsrMtx->nrows % row_block_size;

    bdbcsrMtx->nrows = bdcsrMtx->nrows;
    bdbcsrMtx->ncols = bdcsrMtx->ncols;
    bdbcsrMtx->nnz = bdcsrMtx->nnz;
    bdbcsrMtx->npartitions = bdcsrMtx->npartitions; // Equal with vert_partitions
    bdbcsrMtx->horz_partitions = bdcsrMtx->horz_partitions;
    bdbcsrMtx->vert_partitions = bdcsrMtx->vert_partitions;
    bdbcsrMtx->row_block_size = row_block_size; 
    bdbcsrMtx->col_block_size = col_block_size;
    bdbcsrMtx->num_block_rows = num_block_rows;
    bdbcsrMtx->num_block_cols = num_block_cols;
    bdbcsrMtx->num_rows_left = num_rows_left;
    bdbcsrMtx->max_tile_width = bdcsrMtx->max_tile_width; 

    bdbcsrMtx->vert_tile_widths = (uint32_t *) calloc((bdbcsrMtx->vert_partitions + 1), sizeof(uint32_t));
    bdbcsrMtx->nnzs_per_vert_partition = (uint32_t *) calloc(bdbcsrMtx->vert_partitions, sizeof(uint32_t));
    bdbcsrMtx->nnzs_per_column = (uint32_t *) calloc((bdbcsrMtx->ncols), sizeof(uint32_t));

    memcpy(bdbcsrMtx->vert_tile_widths, bdcsrMtx->vert_tile_widths, (bdcsrMtx->vert_partitions + 1) * sizeof(uint32_t));
    memcpy(bdbcsrMtx->nnzs_per_vert_partition, bdcsrMtx->nnzs_per_vert_partition, bdcsrMtx->vert_partitions * sizeof(uint32_t));
    memcpy(bdbcsrMtx->nnzs_per_column, bdcsrMtx->nnzs_per_column, bdcsrMtx->ncols * sizeof(uint32_t));


    uint32_t *bAp;
    uint32_t *bAj;
    val_dt *bAx;

    //tmp variables
    uint32_t *block_count;
    uint32_t *bAp_next;
    uint64_t I, J;
    uint64_t i, j, k, j0, di;
    val_dt a_ij;

    bAp = (uint32_t *) malloc(bdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    bAp_next = (uint32_t *) malloc(bdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    block_count = (uint32_t *) malloc((num_block_cols) * sizeof(uint32_t));
    memset(block_count, 0, num_block_cols * sizeof(uint32_t));
    bdbcsrMtx->blocks_per_vert_partition = (uint32_t *) calloc(bdbcsrMtx->vert_partitions, sizeof(uint32_t));

    //Phase I: Count the exact number of new blocks to create.
    uint64_t total_nnzs = 0;
    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < bdbcsrMtx->vert_partitions; c++) {

        uint32_t num_blocks = 0;
        uint32_t partition = c;
        uint32_t ptr_offset_bdcsr = c * (bdcsrMtx->nrows + 1);
        uint32_t ptr_offset = c * (num_block_rows + 1);

        bAp[ptr_offset] = 0; 
        if(num_rows_left == 0) {    
            for(I=0; I<num_block_rows; I++) {    
                for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {    
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr+i+1]; k++) {    
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j/col_block_size;
                        if(block_count[J] == 0) {    
                            num_blocks++;
                            block_count[J]++;
                        }
                    }
                }
                bAp[ptr_offset + I+1] = num_blocks;
                for(i = 0; i < num_block_cols; i++)
                    block_count[i] = 0;
            }

        } else {
            for(I=0; I<num_block_rows-1; I++) {
                for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j/col_block_size;
                        if(block_count[J] == 0) {
                            num_blocks++;
                            block_count[J]++;
                        }
                    }
                }
                bAp[ptr_offset + I+1] = num_blocks;
                for(i = 0; i < num_block_cols; i++)
                    block_count[i] = 0;
            }
            for(i = (num_block_rows-1) * row_block_size; i < ((num_block_rows-1) * row_block_size + num_rows_left); i++) {
                for (k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                    j = bdcsrMtx->dcolind[total_nnzs + k];
                    J = j/col_block_size;
                    if (block_count[J] == 0) {
                        num_blocks++;
                        block_count[J]++;
                    }
                }
            }
            bAp[ptr_offset + num_block_rows] = num_blocks;
            for(i = 0; i < num_block_cols; i++)
                block_count[i] = 0;
        }

        total_nnzs += bdcsrMtx->nnzs_per_vert_partition[partition];
        bdbcsrMtx->blocks_per_vert_partition[partition] = num_blocks;
        total_blocks += num_blocks;
    }


    bdbcsrMtx->nblocks = total_blocks;
    bAj = (uint32_t *) malloc((total_blocks + 1) * sizeof(uint32_t));
    bAx = (val_dt *) calloc(((total_blocks + 1) * row_block_size * col_block_size), sizeof(val_dt));
    val_dt *blocks = (val_dt *) malloc((row_block_size * col_block_size * num_block_cols) * sizeof(val_dt));
    memset(blocks, 0, row_block_size * col_block_size * num_block_cols * sizeof(val_dt));
    bdbcsrMtx->nnz_per_block = (uint32_t *) calloc(total_blocks, sizeof(uint32_t));
    memcpy(bAp_next, bAp, bdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));

    total_nnzs = 0;
    total_blocks = 0;
    for (uint32_t c = 0; c < bdcsrMtx->vert_partitions; c++) {

        uint32_t partition = c;
        uint32_t ptr_offset_bdcsr = c * (bdcsrMtx->nrows + 1);
        uint32_t ptr_offset = c * (num_block_rows + 1);


        //Phase II: Copy all blocks.
        if(num_rows_left == 0) {
            for(I=0; I < num_block_rows; I++) {
                for(i = I * row_block_size, di=0; di<row_block_size; di++, i++) {
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;
                        a_ij = bdcsrMtx->dval[total_nnzs + k];
                        blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                        block_count[J]++;
                    }
                }
                for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size; 

                        if(block_count[J] > 0) {
                            uint64_t k_next = bAp_next[ptr_offset + I]; 
                            bAj[total_blocks + k_next] = J; 
                            memcpy(bAx + total_blocks * row_block_size * col_block_size + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                            bAp_next[ptr_offset + I]++;
                            assert(bAp_next[ptr_offset + I] <= bAp[ptr_offset + I+1]);
                            block_count[J] = 0;
                            memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                        }
                    }
                }
            }
        } else {
            for(I = 0; I < num_block_rows-1; I++) {
                for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k<bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;
                        a_ij = bdcsrMtx->dval[total_nnzs + k];
                        blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                        block_count[J]++;
                    }
                }

                for(i = I*row_block_size, di=0; di < row_block_size; di++, i++) {
                    for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr+i+1]; k++) {
                        j = bdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;

                        if (block_count[J] > 0) {
                            uint64_t k_next = bAp_next[ptr_offset + I];
                            bAj[total_blocks + k_next] = J;
                            memcpy(bAx + total_blocks * row_block_size * col_block_size + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                            bAp_next[ptr_offset + I]++;
                            assert(bAp_next[ptr_offset + I] <= bAp[ptr_offset + I+1]);
                            block_count[J] = 0;
                            memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                        }
                    }
                }

            }

            for(i = (num_block_rows-1)*row_block_size, di=0; di < num_rows_left; di++, i++) {
                for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k < bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                    j = bdcsrMtx->dcolind[total_nnzs + k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;
                    a_ij = bdcsrMtx->dval[total_nnzs + k];
                    blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                    block_count[J]++;
                }
            }

            for(i = (num_block_rows-1)*row_block_size, di=0; di<num_rows_left; di++, i++) {
                for(k = bdcsrMtx->drowptr[ptr_offset_bdcsr + i]; k<bdcsrMtx->drowptr[ptr_offset_bdcsr + i+1]; k++) {
                    j = bdcsrMtx->dcolind[total_nnzs + k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;

                    if(block_count[J] > 0) {
                        uint64_t k_next = bAp_next[ptr_offset + num_block_rows-1];
                        bAj[total_blocks + k_next] = J; 
                        memcpy(bAx + total_blocks * row_block_size * col_block_size + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                        bAp_next[ptr_offset + num_block_rows-1]++;
                        assert(bAp_next[ptr_offset + num_block_rows-1] <= bAp[ptr_offset + num_block_rows]);
                        block_count[J] = 0;
                        memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                    }
                }
            }

        }

        total_nnzs += bdcsrMtx->nnzs_per_vert_partition[partition];
        total_blocks += bdbcsrMtx->blocks_per_vert_partition[partition];
    }


    free(block_count);
    free(blocks);
    free(bAp_next);

    bdbcsrMtx->browptr = bAp;
    bdbcsrMtx->bcolind = bAj;
    bdbcsrMtx->bval = bAx;

    return bdbcsrMtx;
}

/**
 * @brief count nnz per block in BDBCSR format matrix
 */
void countNNZperBlockBDBCSRMatrix(struct BDBCSRMatrix *A) {
    uint64_t total_nnzs = 0;
    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < A->vert_partitions; c++) {
        uint32_t ptr_offset = c * (A->num_block_rows + 1);
        for(uint32_t n=0; n<A->num_block_rows; n++) {
            for(uint32_t i=A->browptr[ptr_offset + n]; i< A->browptr[ptr_offset + n+1]; i++){
                //uint32_t j = A->bcolind[total_blocks + i];
                for(uint32_t blr=0; blr < A->row_block_size; blr++){
                    for(uint32_t blc=0; blc < A->col_block_size; blc++) {
                        if (A->bval[(total_blocks + i) * A->row_block_size * A->col_block_size + blr * A->col_block_size + blc] != 0){
                            A->nnz_per_block[total_blocks + i]++;
                            total_nnzs++;
                        }
                    }
                }
            }
        }

        total_blocks += A->blocks_per_vert_partition[c];
    }
    assert(total_nnzs == A->nnz && "#nnzs do not match!");

    total_nnzs = 0;
    for(uint32_t b = 0; b < A->nblocks; b++) {
        total_nnzs += A->nnz_per_block[b];
    }
    assert(total_nnzs == A->nnz && "#nnzs do not match!");
}


/**
 * @brief deallocate matrix in BDBCSR format 
 * @param matrix in BDBCSR format
 */ 
void freeBDBCSRMatrix(struct BDBCSRMatrix *bdbcsrMtx) {
    free(bdbcsrMtx->vert_tile_widths);
    free(bdbcsrMtx->nnzs_per_column);
    free(bdbcsrMtx->nnzs_per_vert_partition);
    free(bdbcsrMtx->blocks_per_vert_partition);
    free(bdbcsrMtx->nnz_per_block);
    free(bdbcsrMtx->browptr);
    free(bdbcsrMtx->bcolind);
    free(bdbcsrMtx->bval);
    free(bdbcsrMtx);
}



#endif
