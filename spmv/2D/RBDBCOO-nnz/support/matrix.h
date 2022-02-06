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
 * @brief RBDCSR matrix format 
 * 2D-partitioned matrix with equally-wide vertical tiles and CSR on each vertical tile
 */
struct RBDCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t tile_width;
    uint32_t *nnzs_per_vert_partition;
    uint32_t *drowptr;
    uint32_t *dcolind;
    val_dt *dval;
};


/**
 * @brief RBDBCSR matrix format 
 * 2D-partitioned matrix with equally-wide vertical tiles and BCSR on each vertical tile
 */
struct RBDBCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t nblocks;
    uint32_t tile_width;
    uint32_t *nnzs_per_vert_partition;
    uint32_t *blocks_per_vert_partition;
    uint64_t row_block_size;
    uint64_t col_block_size;
    uint32_t num_block_rows;
    uint32_t num_block_cols;
    uint32_t num_rows_left;
    uint32_t* browptr;  // row pointer
    uint32_t* bcolind;  // column indices
    val_dt* bval;       // nonzeros
    uint32_t* nnz_per_block;  // nnz per block
};

/**
 * @brief RBDBCOO matrix format 
 * 2D-partitioned matrix with equally-wide vertical tiles and BCOO on each vertical tile
 */
struct RBDBCOOMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t nblocks;
    uint32_t tile_width;
    uint32_t *nnzs_per_vert_partition;
    uint32_t *blocks_per_vert_partition;
    uint64_t row_block_size;
    uint64_t col_block_size;
    uint32_t num_block_rows;
    uint32_t num_block_cols;
    uint32_t num_rows_left;
    struct bind_t* bind;  // row-column indices
    val_dt* bval;       // nonzeros
    uint32_t* nnz_per_block;  // nnz per block
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
 * @brief convert matrix from COO to RBDCSR format 
 * @param matrix in COO format
 * @param horz_partitions, vert_partitions
 */
struct RBDCSRMatrix *coo2rbdcsr(struct COOMatrix *cooMtx, int horz_partitions, int vert_partitions) {

    struct RBDCSRMatrix *rbdcsrMtx;
    rbdcsrMtx = (struct RBDCSRMatrix *) malloc(sizeof(struct RBDCSRMatrix));

    rbdcsrMtx->nrows = cooMtx->nrows;
    rbdcsrMtx->ncols = cooMtx->ncols;
    rbdcsrMtx->nnz = cooMtx->nnz;

    rbdcsrMtx->npartitions = vert_partitions;
    rbdcsrMtx->horz_partitions = horz_partitions;
    rbdcsrMtx->vert_partitions = vert_partitions;
    rbdcsrMtx->tile_width = cooMtx->ncols / vert_partitions;
    if (cooMtx->ncols % vert_partitions != 0)
        rbdcsrMtx->tile_width++;

    // Count nnzs per partition
    rbdcsrMtx->nnzs_per_vert_partition = (uint32_t *) calloc((rbdcsrMtx->npartitions), sizeof(uint32_t));

    uint32_t p_row, p_col, local_col;
    rbdcsrMtx->drowptr = (uint32_t *) calloc((rbdcsrMtx->nrows + 2) * rbdcsrMtx->npartitions, sizeof(uint32_t));
    rbdcsrMtx->dcolind = (uint32_t *) malloc((rbdcsrMtx->nnz + 2) * sizeof(uint32_t));
    rbdcsrMtx->dval = (val_dt *) calloc((rbdcsrMtx->nnz + 8),  sizeof(val_dt)); // Padding needed

    for (uint32_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind; 
        p_col = cooMtx->nnzs[n].colind / rbdcsrMtx->tile_width;
        rbdcsrMtx->nnzs_per_vert_partition[p_col]++;
        rbdcsrMtx->drowptr[p_col * (rbdcsrMtx->nrows + 1) + p_row]++;
    }

    // Normalize local rowptrs within each partition 
    for (uint32_t p = 0; p < rbdcsrMtx->npartitions; p++) {
        uint32_t sumBeforeNextRow = 0;
        for(unsigned int rowIndx = 0; rowIndx < rbdcsrMtx->nrows; ++rowIndx) {
            uint32_t sumBeforeRow = sumBeforeNextRow;
            sumBeforeNextRow += rbdcsrMtx->drowptr[p * (rbdcsrMtx->nrows + 1) + rowIndx];
            rbdcsrMtx->drowptr[p * (rbdcsrMtx->nrows + 1) + rowIndx] = sumBeforeRow;
        }
        rbdcsrMtx->drowptr[p * (rbdcsrMtx->nrows + 1) + rbdcsrMtx->nrows] = sumBeforeNextRow;
    }

    // Fill nnzs to partitions
    uint64_t *global_nnzs = (uint64_t *) calloc(rbdcsrMtx->npartitions, sizeof(uint64_t));
    uint64_t *local_nnzs = (uint64_t *) calloc(rbdcsrMtx->npartitions, sizeof(uint64_t));
    uint64_t total_nnzs = 0;
    for (uint32_t p = 0; p < rbdcsrMtx->npartitions; p++) {
        global_nnzs[p] = total_nnzs;
        total_nnzs += rbdcsrMtx->nnzs_per_vert_partition[p];
    }

    for (uint64_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind; 
        p_col = cooMtx->nnzs[n].colind / rbdcsrMtx->tile_width;
        local_col =  cooMtx->nnzs[n].colind - (p_col * rbdcsrMtx->tile_width); 
        rbdcsrMtx->dcolind[global_nnzs[p_col] + local_nnzs[p_col]] = local_col; 
        rbdcsrMtx->dval[global_nnzs[p_col] + local_nnzs[p_col]] = cooMtx->nnzs[n].val;
        local_nnzs[p_col]++;
    }

    free(global_nnzs);
    free(local_nnzs);

    return rbdcsrMtx;
}

/**
 * @brief deallocate matrix in RBDCSR format 
 * @param matrix in RBDCSR format
 */ 
void freeRBDCSRMatrix(struct RBDCSRMatrix *rbdcsrMtx) {
    free(rbdcsrMtx->nnzs_per_vert_partition);
    free(rbdcsrMtx->drowptr);
    free(rbdcsrMtx->dcolind);
    free(rbdcsrMtx->dval);
    free(rbdcsrMtx);
}


/**
 * @brief convert matrix from RBDCSR to RBDBCSR format 
 * @param matrix in RBDCSR format
 * @param horz_partitions, vert_partitions, row_block_size, col_block_size
 */
struct RBDBCSRMatrix *rbdcsr2rbdbcsr(struct RBDCSRMatrix *rbdcsrMtx, uint32_t row_block_size, uint32_t col_block_size) {
    struct RBDBCSRMatrix *rbdbcsrMtx;
    rbdbcsrMtx = (struct RBDBCSRMatrix *) malloc(sizeof(struct RBDBCSRMatrix));

    uint32_t num_block_rows = (rbdcsrMtx->nrows + row_block_size - 1) / row_block_size;
    uint32_t num_block_cols = (rbdcsrMtx->tile_width + col_block_size - 1) / col_block_size; 
    uint32_t num_rows_left = rbdcsrMtx->nrows % row_block_size;

    rbdbcsrMtx->nrows = rbdcsrMtx->nrows;
    rbdbcsrMtx->ncols = rbdcsrMtx->ncols;
    rbdbcsrMtx->nnz = rbdcsrMtx->nnz;
    rbdbcsrMtx->npartitions = rbdcsrMtx->npartitions; 
    rbdbcsrMtx->horz_partitions = rbdcsrMtx->horz_partitions;
    rbdbcsrMtx->vert_partitions = rbdcsrMtx->vert_partitions;
    rbdbcsrMtx->row_block_size = row_block_size; 
    rbdbcsrMtx->col_block_size = col_block_size;
    rbdbcsrMtx->num_block_rows = num_block_rows;
    rbdbcsrMtx->num_block_cols = num_block_cols;
    rbdbcsrMtx->num_rows_left = num_rows_left;
    rbdbcsrMtx->tile_width = rbdcsrMtx->tile_width; 

    rbdbcsrMtx->nnzs_per_vert_partition = (uint32_t *) calloc(rbdbcsrMtx->vert_partitions, sizeof(uint32_t));
    memcpy(rbdbcsrMtx->nnzs_per_vert_partition, rbdcsrMtx->nnzs_per_vert_partition, rbdcsrMtx->vert_partitions * sizeof(uint32_t));


    uint32_t *bAp;
    uint32_t *bAj;
    val_dt *bAx;

    //tmp variables
    uint32_t *block_count;
    uint32_t *bAp_next;
    uint64_t I, J;
    uint64_t i, j, k, j0, di;
    val_dt a_ij;

    bAp = (uint32_t *) malloc(rbdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    bAp_next = (uint32_t *) malloc(rbdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    block_count = (uint32_t *) malloc((num_block_cols) * sizeof(uint32_t));
    memset(block_count, 0, num_block_cols * sizeof(uint32_t));
    rbdbcsrMtx->blocks_per_vert_partition = (uint32_t *) calloc(rbdbcsrMtx->vert_partitions, sizeof(uint32_t));

    //Phase I: Count the exact number of new blocks to create.
    uint64_t total_nnzs = 0;
    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < rbdbcsrMtx->vert_partitions; c++) {
        uint32_t num_blocks = 0;
        uint32_t partition = c;
        uint32_t ptr_offset_rbdcsr = c * (rbdcsrMtx->nrows + 1);
        uint32_t ptr_offset = c * (num_block_rows + 1);

        bAp[ptr_offset] = 0; 
        if(num_rows_left == 0) {    
            for(I=0; I<num_block_rows; I++) {    
                for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {    
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr+i+1]; k++) {    
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
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
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
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
                for (k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                    j = rbdcsrMtx->dcolind[total_nnzs + k];
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

        total_nnzs += rbdcsrMtx->nnzs_per_vert_partition[partition];
        rbdbcsrMtx->blocks_per_vert_partition[partition] = num_blocks;
        total_blocks += num_blocks;
    }


    rbdbcsrMtx->nblocks = total_blocks;
    bAj = (uint32_t *) malloc((total_blocks + 1) * sizeof(uint32_t));
    bAx = (val_dt *) calloc(((total_blocks + 1) * row_block_size * col_block_size), sizeof(val_dt));
    val_dt *blocks = (val_dt *) malloc((row_block_size * col_block_size * num_block_cols) * sizeof(val_dt));
    memset(blocks, 0, row_block_size * col_block_size * num_block_cols * sizeof(val_dt));
    rbdbcsrMtx->nnz_per_block = (uint32_t *) calloc(total_blocks, sizeof(uint32_t));
    memcpy(bAp_next, bAp, rbdbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));

    total_nnzs = 0;
    total_blocks = 0;
    for (uint32_t c = 0; c < rbdcsrMtx->vert_partitions; c++) {
        uint32_t partition = c;
        uint32_t ptr_offset_rbdcsr = c * (rbdcsrMtx->nrows + 1);
        uint32_t ptr_offset = c * (num_block_rows + 1);


        //Phase II: Copy all blocks.
        if(num_rows_left == 0) {
            for(I=0; I < num_block_rows; I++) {
                for(i = I * row_block_size, di=0; di<row_block_size; di++, i++) {
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;
                        a_ij = rbdcsrMtx->dval[total_nnzs + k];
                        blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                        block_count[J]++;
                    }
                }
                for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
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
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k<rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;
                        a_ij = rbdcsrMtx->dval[total_nnzs + k];
                        blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                        block_count[J]++;
                    }
                }

                for(i = I*row_block_size, di=0; di < row_block_size; di++, i++) {
                    for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr+i+1]; k++) {
                        j = rbdcsrMtx->dcolind[total_nnzs + k];
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
                for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k < rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                    j = rbdcsrMtx->dcolind[total_nnzs + k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;
                    a_ij = rbdcsrMtx->dval[total_nnzs + k];
                    blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                    block_count[J]++;
                }
            }

            for(i = (num_block_rows-1)*row_block_size, di=0; di<num_rows_left; di++, i++) {
                for(k = rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i]; k<rbdcsrMtx->drowptr[ptr_offset_rbdcsr + i+1]; k++) {
                    j = rbdcsrMtx->dcolind[total_nnzs + k];
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


        total_nnzs += rbdcsrMtx->nnzs_per_vert_partition[partition];
        total_blocks += rbdbcsrMtx->blocks_per_vert_partition[partition];
    }


    free(block_count);
    free(blocks);
    free(bAp_next);

    rbdbcsrMtx->browptr = bAp;
    rbdbcsrMtx->bcolind = bAj;
    rbdbcsrMtx->bval = bAx;

    return rbdbcsrMtx;
}

/**
 * @brief count nnz per block in DBCSR format matrix
 */
void countNNZperBlockRBDBCSRMatrix(struct RBDBCSRMatrix *A) {
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
 * @brief partition function for quickSort 
 */
int32_t partitionRBDBCSRMatrix(struct RBDBCSRMatrix *rbdbcsrMtx, uint32_t low, uint32_t high, uint64_t total_blocks) {

    uint32_t pivot = rbdbcsrMtx->bcolind[total_blocks + high]; 
    int32_t i = low - 1;
    uint32_t temp_ind;
    val_dt temp_val;
    
    for(uint32_t j = low; j <= high - 1; j++) {
        if(rbdbcsrMtx->bcolind[total_blocks + j] < pivot) {
            i++;
            // swap(i, j)
            temp_ind = rbdbcsrMtx->bcolind[total_blocks + i];
            rbdbcsrMtx->bcolind[total_blocks + i] = rbdbcsrMtx->bcolind[total_blocks + j];
            rbdbcsrMtx->bcolind[total_blocks + j] = temp_ind;
            for(uint32_t r=0; r<rbdbcsrMtx->row_block_size; r++) {
                for(uint32_t c=0; c<rbdbcsrMtx->col_block_size; c++) {
                    temp_val = rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + i * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c];
                    rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + i * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c] = rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + j * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c];
                    rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + j * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c] = temp_val;
                }
            }
        }
    }

    // swap(i+1, high)
    temp_ind = rbdbcsrMtx->bcolind[total_blocks + i+1];
    rbdbcsrMtx->bcolind[total_blocks + i+1] = rbdbcsrMtx->bcolind[total_blocks + high];
    rbdbcsrMtx->bcolind[total_blocks + high] = temp_ind;
    for(uint32_t r=0; r<rbdbcsrMtx->row_block_size; r++) {
        for(uint32_t c=0; c<rbdbcsrMtx->col_block_size; c++) {
            temp_val = rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + (i+1) * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c];
            rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + (i+1) * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c] = rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + high * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c];
            rbdbcsrMtx->bval[total_blocks * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + high * rbdbcsrMtx->row_block_size * rbdbcsrMtx->col_block_size + r * rbdbcsrMtx->col_block_size + c] = temp_val;
        }
    }
    return (i+1); 
}

/**
 * @brief quickSort for matrix in RBDBCSR format
 */
void quickSortRBDBCSRMatrix(struct RBDBCSRMatrix *rbdbcsrMtx, int32_t low, int32_t high, uint32_t total_blocks) {

    if (low < high) {
        int32_t mid = partitionRBDBCSRMatrix(rbdbcsrMtx, low, high, total_blocks);

        quickSortRBDBCSRMatrix(rbdbcsrMtx, low, mid - 1, total_blocks); 
        quickSortRBDBCSRMatrix(rbdbcsrMtx, mid + 1, high, total_blocks); 
    }
}

/**
 * @brief Sort wrapper for matrix in RBDBCSR format
 */
void sortRBDBCSRMatrix(struct RBDBCSRMatrix *rbdbcsrMtx) {

    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < rbdbcsrMtx->vert_partitions; c++) {
        uint32_t partition = c;
        uint32_t ptr_offset = partition * (rbdbcsrMtx->num_block_rows + 1);

        for(uint32_t n=0; n<rbdbcsrMtx->num_block_rows; n++) {
            int32_t low = rbdbcsrMtx->browptr[ptr_offset + n]; 
            int32_t high = rbdbcsrMtx->browptr[ptr_offset + n+1] - 1; 

            quickSortRBDBCSRMatrix(rbdbcsrMtx, low, high, total_blocks);
        }
        total_blocks += rbdbcsrMtx->blocks_per_vert_partition[partition];
    }
}


/**
 * @brief deallocate matrix in RBDBCSR format 
 * @param matrix in RBDBCSR format
 */ 
void freeRBDBCSRMatrix(struct RBDBCSRMatrix *rbdbcsrMtx) {
    free(rbdbcsrMtx->nnzs_per_vert_partition);
    free(rbdbcsrMtx->blocks_per_vert_partition);
    free(rbdbcsrMtx->nnz_per_block);
    free(rbdbcsrMtx->browptr);
    free(rbdbcsrMtx->bcolind);
    free(rbdbcsrMtx->bval);
    free(rbdbcsrMtx);
}

/**
 * @brief convert matrix from RBDBCSR to RBDBCOO format 
 * @param matrix in RBDBCSR format
 */ 
struct RBDBCOOMatrix *rbdbcsr2rbdbcoo(struct RBDBCSRMatrix *rbdbcsrMtx) {
    struct RBDBCOOMatrix *rbdbcooMtx;
    rbdbcooMtx = (struct RBDBCOOMatrix *) malloc(sizeof(struct RBDBCOOMatrix));

    rbdbcooMtx->nrows = rbdbcsrMtx->nrows;
    rbdbcooMtx->ncols = rbdbcsrMtx->ncols;
    rbdbcooMtx->nnz = rbdbcsrMtx->nnz;
    rbdbcooMtx->nblocks = rbdbcsrMtx->nblocks;
    rbdbcooMtx->npartitions = rbdbcsrMtx->npartitions; 
    rbdbcooMtx->horz_partitions = rbdbcsrMtx->horz_partitions;
    rbdbcooMtx->vert_partitions = rbdbcsrMtx->vert_partitions;
    rbdbcooMtx->tile_width = rbdbcsrMtx->tile_width;
    rbdbcooMtx->num_block_rows = rbdbcsrMtx->num_block_rows;
    rbdbcooMtx->num_block_cols = rbdbcsrMtx->num_block_cols;
    rbdbcooMtx->num_rows_left = rbdbcsrMtx->num_rows_left;
    rbdbcooMtx->row_block_size = rbdbcsrMtx->row_block_size;
    rbdbcooMtx->col_block_size = rbdbcsrMtx->col_block_size;


    rbdbcooMtx->nnzs_per_vert_partition = (uint32_t *) malloc(rbdbcooMtx->vert_partitions * sizeof(uint32_t));
    rbdbcooMtx->blocks_per_vert_partition = (uint32_t *) malloc(rbdbcooMtx->vert_partitions * sizeof(uint32_t));
    rbdbcooMtx->bind = (struct bind_t *) malloc(rbdbcooMtx->nblocks * sizeof(struct bind_t));
    rbdbcooMtx->bval = (val_dt *) malloc((rbdbcooMtx->nblocks + 1) * rbdbcooMtx->row_block_size * rbdbcooMtx->col_block_size * sizeof(val_dt));
    rbdbcooMtx->nnz_per_block = (uint32_t *) malloc(rbdbcooMtx->nblocks * sizeof(uint32_t));

    memcpy(rbdbcooMtx->nnzs_per_vert_partition, rbdbcsrMtx->nnzs_per_vert_partition, rbdbcsrMtx->vert_partitions * sizeof(uint32_t));
    memcpy(rbdbcooMtx->blocks_per_vert_partition, rbdbcsrMtx->blocks_per_vert_partition, rbdbcsrMtx->vert_partitions * sizeof(uint32_t));
    memcpy(rbdbcooMtx->nnz_per_block, rbdbcsrMtx->nnz_per_block, rbdbcsrMtx->nblocks * sizeof(uint32_t));


    uint64_t total_blocks = 0;
    for (uint32_t c = 0; c < rbdbcsrMtx->vert_partitions; c++) {
        uint32_t partition = c;
        uint32_t ptr_offset = partition * (rbdbcsrMtx->num_block_rows + 1);

        for(uint64_t n=0; n<rbdbcsrMtx->num_block_rows; n++) {
            for(uint64_t i=rbdbcsrMtx->browptr[ptr_offset + n]; i<rbdbcsrMtx->browptr[ptr_offset + n+1]; i++){
                rbdbcooMtx->bind[total_blocks + i].rowind = n;
                rbdbcooMtx->bind[total_blocks + i].colind = rbdbcsrMtx->bcolind[total_blocks + i];
            }
        }
        total_blocks += rbdbcsrMtx->blocks_per_vert_partition[partition];
    }

    memcpy(rbdbcooMtx->bval, rbdbcsrMtx->bval, (rbdbcooMtx->nblocks + 1) * rbdbcooMtx->row_block_size * rbdbcooMtx->col_block_size * sizeof(val_dt));

    return rbdbcooMtx;
}


/**
 * @brief deallocate matrix in RBDBCOO format 
 * @param matrix in RBDBCOO format
 */  
void freeRBDBCOOMatrix(struct RBDBCOOMatrix *rbdbcooMtx) {
    free(rbdbcooMtx->nnzs_per_vert_partition);
    free(rbdbcooMtx->blocks_per_vert_partition);
    free(rbdbcooMtx->bind);
    free(rbdbcooMtx->bval);
    free(rbdbcooMtx->nnz_per_block);
    free(rbdbcooMtx);
}


#endif
