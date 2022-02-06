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
 * @brief DCSR matrix format 
 * 2D-partitioned matrix with equally-sized tiles and CSR on each tile
 */
struct DCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t tile_height;
    uint32_t tile_width; 
    uint32_t *nnzs_per_partition;
    uint32_t *drowptr;
    uint32_t *dcolind;
    val_dt *dval;
};

/**
 * @brief DBCSR matrix format 
 * 2D-partitioned matrix with equally-sized tiles and BCSR on each tile
 */
struct DBCSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t nblocks;
    uint64_t tile_height;
    uint64_t tile_width;
    uint32_t *nnzs_per_partition;
    uint32_t *blocks_per_partition;
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
 * @brief DBCOO matrix format 
 * 2D-partitioned matrix with equally-sized tiles and BCOO on each tile
 */
struct DBCOOMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t npartitions;
    uint32_t horz_partitions;
    uint32_t vert_partitions;
    uint32_t nblocks;
    uint64_t tile_height;
    uint64_t tile_width;
    uint32_t *nnzs_per_partition;
    uint32_t *blocks_per_partition;
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
 * @brief convert matrix from COO to DCSR format 
 * @param matrix in COO format
 * @param horz_partitions, vert_partitions
 */
struct DCSRMatrix *coo2dcsr(struct COOMatrix *cooMtx, int horz_partitions, int vert_partitions) {

    struct DCSRMatrix *dcsrMtx;
    dcsrMtx = (struct DCSRMatrix *) malloc(sizeof(struct DCSRMatrix));

    dcsrMtx->nrows = cooMtx->nrows;
    dcsrMtx->ncols = cooMtx->ncols;
    dcsrMtx->nnz = cooMtx->nnz;

    dcsrMtx->npartitions = horz_partitions * vert_partitions;
    dcsrMtx->horz_partitions = horz_partitions;
    dcsrMtx->vert_partitions = vert_partitions;
    dcsrMtx->tile_height = cooMtx->nrows / horz_partitions;
    if (cooMtx->nrows % horz_partitions != 0)
        dcsrMtx->tile_height++;
    dcsrMtx->tile_width = cooMtx->ncols / vert_partitions;
    if (cooMtx->ncols % vert_partitions != 0)
        dcsrMtx->tile_width++;

    dcsrMtx->nnzs_per_partition = (uint32_t *) calloc((horz_partitions * vert_partitions), sizeof(uint32_t));
    dcsrMtx->drowptr = (uint32_t *) calloc((dcsrMtx->tile_height+2) * dcsrMtx->npartitions, sizeof(uint32_t));
    dcsrMtx->dcolind = (uint32_t *) malloc((dcsrMtx->nnz + 2) * sizeof(uint32_t));
    dcsrMtx->dval = (val_dt *) calloc((dcsrMtx->nnz + 8),  sizeof(val_dt)); // Padding needed

    // Count nnzs for each partition
    uint32_t p_row, local_row;
    uint32_t p_col, local_col;
    uint32_t p;
    for (uint32_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind / dcsrMtx->tile_height; 
        p_col = cooMtx->nnzs[n].colind / dcsrMtx->tile_width; 
        p = p_row * vert_partitions + p_col;
        dcsrMtx->nnzs_per_partition[p]++;
        local_row = cooMtx->nnzs[n].rowind - (p_row * dcsrMtx->tile_height); 
        dcsrMtx->drowptr[p * (dcsrMtx->tile_height + 1) + local_row]++;
    }

    // Normalize local rowptrs within each partition 
    for (uint32_t p = 0; p < dcsrMtx->npartitions; p++) {
        uint32_t sumBeforeNextRow = 0;
        for(unsigned int rowIndx = 0; rowIndx < dcsrMtx->tile_height; ++rowIndx) {
            uint32_t sumBeforeRow = sumBeforeNextRow;
            sumBeforeNextRow += dcsrMtx->drowptr[p * (dcsrMtx->tile_height + 1) + rowIndx];
            dcsrMtx->drowptr[p * (dcsrMtx->tile_height + 1) + rowIndx] = sumBeforeRow;
        }
        dcsrMtx->drowptr[p * (dcsrMtx->tile_height + 1) + dcsrMtx->tile_height] = sumBeforeNextRow;
    }

    // Fill nnzs to partitions
    uint64_t *global_nnzs = (uint64_t *) calloc(dcsrMtx->npartitions, sizeof(uint64_t));
    uint64_t *local_nnzs = (uint64_t *) calloc(dcsrMtx->npartitions, sizeof(uint64_t));
    uint64_t total_nnzs = 0;
    for (uint32_t p = 0; p < dcsrMtx->npartitions; p++) {
        global_nnzs[p] = total_nnzs;
        total_nnzs += dcsrMtx->nnzs_per_partition[p];
    }

    for (uint64_t n = 0; n < cooMtx->nnz; n++) {
        p_row = cooMtx->nnzs[n].rowind / dcsrMtx->tile_height; 
        p_col = cooMtx->nnzs[n].colind / dcsrMtx->tile_width; 
        p = p_row * vert_partitions + p_col;
        local_col = cooMtx->nnzs[n].colind - (p_col * dcsrMtx->tile_width); 
        dcsrMtx->dcolind[global_nnzs[p] + local_nnzs[p]] = local_col; 
        dcsrMtx->dval[global_nnzs[p] + local_nnzs[p]] = cooMtx->nnzs[n].val;
        local_nnzs[p]++;
    }

    free(global_nnzs);
    free(local_nnzs);
    return dcsrMtx;

}

/**
 * @brief deallocate matrix in DCSR format 
 * @param matrix in DCSR format
 */ 
void freeDCSRMatrix(struct DCSRMatrix *dcsrMtx) {
    free(dcsrMtx->nnzs_per_partition);
    free(dcsrMtx->drowptr);
    free(dcsrMtx->dcolind);
    free(dcsrMtx->dval);
    free(dcsrMtx);
}

/**
 * @brief convert matrix from DCSR to DBCSR format 
 * @param matrix in DCSR format
 * @param block size dimensions
 */
struct DBCSRMatrix *dcsr2dbcsr(struct DCSRMatrix *dcsrMtx, uint32_t row_block_size, uint32_t col_block_size) {
    struct DBCSRMatrix *dbcsrMtx;
    dbcsrMtx = (struct DBCSRMatrix *) malloc(sizeof(struct DBCSRMatrix));

    uint32_t num_block_rows = (dcsrMtx->tile_height + row_block_size - 1) / row_block_size;
    uint32_t num_block_cols = (dcsrMtx->tile_width + col_block_size - 1) / col_block_size;
    uint32_t num_rows_left = dcsrMtx->tile_height % row_block_size;

    dbcsrMtx->nrows = dcsrMtx->nrows;
    dbcsrMtx->ncols = dcsrMtx->ncols;
    dbcsrMtx->nnz = dcsrMtx->nnz;
    dbcsrMtx->npartitions = dcsrMtx->npartitions;
    dbcsrMtx->horz_partitions = dcsrMtx->horz_partitions;
    dbcsrMtx->vert_partitions = dcsrMtx->vert_partitions;
    dbcsrMtx->tile_height = dcsrMtx->tile_height; 
    dbcsrMtx->tile_width = dcsrMtx->tile_width;
    dbcsrMtx->row_block_size = row_block_size; 
    dbcsrMtx->col_block_size = col_block_size;
    dbcsrMtx->num_block_rows = num_block_rows;
    dbcsrMtx->num_block_cols = num_block_cols;
    dbcsrMtx->num_rows_left = num_rows_left;

    uint32_t *bAp;
    uint32_t *bAj;
    val_dt *bAx;

    //tmp variables
    uint32_t *block_count;
    uint32_t *bAp_next;
    uint64_t I, J;
    uint64_t i, j, k, j0, di;
    val_dt a_ij;

    bAp = (uint32_t *) malloc(dcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    bAp_next = (uint32_t *) malloc(dcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));
    block_count = (uint32_t *) malloc((num_block_cols) * sizeof(uint32_t));
    memset(block_count, 0, num_block_cols * sizeof(uint32_t));

    dbcsrMtx->nnzs_per_partition = (uint32_t *) calloc(dcsrMtx->npartitions, sizeof(uint32_t));
    memcpy(dbcsrMtx->nnzs_per_partition, dcsrMtx->nnzs_per_partition, dcsrMtx->npartitions * sizeof(uint32_t));
    dbcsrMtx->blocks_per_partition = (uint32_t *) calloc(dcsrMtx->npartitions, sizeof(uint32_t));

    //Phase I: Count the exact number of new blocks to create.
    uint64_t total_nnzs = 0;
    uint64_t total_blocks = 0;
    for (uint32_t r = 0; r < dcsrMtx->horz_partitions; r++) {
        for (uint32_t c = 0; c < dcsrMtx->vert_partitions; c++) {

            uint32_t num_blocks = 0;
            uint32_t partition = r * dcsrMtx->vert_partitions + c;
            uint32_t ptr_offset_dcsr = (r * dcsrMtx->vert_partitions + c) * (dcsrMtx->tile_height + 1);
            uint32_t ptr_offset = (r * dcsrMtx->vert_partitions + c) * (num_block_rows + 1);

            bAp[ptr_offset] = 0; 
            if(num_rows_left == 0) {    
                for(I=0; I<num_block_rows; I++) {    
                    for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {    
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr+i+1]; k++) {    
                            j = dcsrMtx->dcolind[total_nnzs + k];
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
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                            j = dcsrMtx->dcolind[total_nnzs + k];
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
                    for (k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                        j = dcsrMtx->dcolind[total_nnzs + k];
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

            total_nnzs += dcsrMtx->nnzs_per_partition[partition];
            dbcsrMtx->blocks_per_partition[partition] = num_blocks;
            total_blocks += num_blocks;
        }
    }


    dbcsrMtx->nblocks = total_blocks;
    bAj = (uint32_t *) malloc((total_blocks + 1) * sizeof(uint32_t));
    bAx = (val_dt *) calloc(((total_blocks + 1) * row_block_size * col_block_size), sizeof(val_dt));
    val_dt *blocks = (val_dt *) malloc((row_block_size * col_block_size * num_block_cols) * sizeof(val_dt));
    memset(blocks, 0, row_block_size * col_block_size * num_block_cols * sizeof(val_dt));
    dbcsrMtx->nnz_per_block = (uint32_t *) calloc(total_blocks, sizeof(uint32_t));
    memcpy(bAp_next, bAp, dbcsrMtx->npartitions * (num_block_rows+2) * sizeof(uint32_t));

    total_nnzs = 0;
    total_blocks = 0;
    for (uint32_t r = 0; r < dcsrMtx->horz_partitions; r++) {
        for (uint32_t c = 0; c < dcsrMtx->vert_partitions; c++) {

            uint32_t partition = r * dcsrMtx->vert_partitions + c;
            uint32_t ptr_offset_dcsr = (r * dcsrMtx->vert_partitions + c) * (dcsrMtx->tile_height + 1);
            uint32_t ptr_offset = (r * dcsrMtx->vert_partitions + c) * (num_block_rows + 1);


            //Phase II: Copy all blocks.
            if(num_rows_left == 0) {
                for(I=0; I < num_block_rows; I++) {
                    for(i = I * row_block_size, di=0; di<row_block_size; di++, i++) {
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                            j = dcsrMtx->dcolind[total_nnzs + k];
                            J = j / col_block_size;
                            j0 = J * col_block_size;
                            a_ij = dcsrMtx->dval[total_nnzs + k];
                            blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                            block_count[J]++;
                        }
                    }
                    for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                            j = dcsrMtx->dcolind[total_nnzs + k];
                            J = j / col_block_size;
                            j0 = J * col_block_size; 

                            if(block_count[J] > 0) {
                                uint64_t k_next = bAp_next[ptr_offset + I]; 
                                //bAi[k_next] = I;  
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
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k<dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                            j = dcsrMtx->dcolind[total_nnzs + k];
                            J = j / col_block_size;
                            j0 = J * col_block_size;
                            a_ij = dcsrMtx->dval[total_nnzs + k];
                            blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                            block_count[J]++;
                        }
                    }

                    for(i = I*row_block_size, di=0; di < row_block_size; di++, i++) {
                        for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr+i+1]; k++) {
                            j = dcsrMtx->dcolind[total_nnzs + k];
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
                    for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k < dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                        j = dcsrMtx->dcolind[total_nnzs + k];
                        J = j / col_block_size;
                        j0 = J * col_block_size;
                        a_ij = dcsrMtx->dval[total_nnzs + k];
                        blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                        block_count[J]++;
                    }
                }

                for(i = (num_block_rows-1)*row_block_size, di=0; di<num_rows_left; di++, i++) {
                    for(k = dcsrMtx->drowptr[ptr_offset_dcsr + i]; k<dcsrMtx->drowptr[ptr_offset_dcsr + i+1]; k++) {
                        j = dcsrMtx->dcolind[total_nnzs + k];
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


            total_nnzs += dcsrMtx->nnzs_per_partition[partition];
            total_blocks += dbcsrMtx->blocks_per_partition[partition];
        }
    }


    free(block_count);
    free(blocks);
    free(bAp_next);

    dbcsrMtx->browptr = bAp;
    dbcsrMtx->bcolind = bAj;
    dbcsrMtx->bval = bAx;

    return dbcsrMtx;
}

/**
 * @brief count nnz per block in DBCSR format matrix
 */
void countNNZperBlockDBCSRMatrix(struct DBCSRMatrix *A) {
    uint64_t total_nnzs = 0;
    uint64_t total_blocks = 0;
    for (uint32_t r = 0; r < A->horz_partitions; r++) {
        for (uint32_t c = 0; c < A->vert_partitions; c++) {

            uint32_t ptr_offset = (r * A->vert_partitions + c) * (A->num_block_rows + 1);
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

            total_blocks += A->blocks_per_partition[r * A->vert_partitions + c];
        }
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
uint32_t partitionDBCSRMatrix(struct DBCSRMatrix *dbcsrMtx, uint32_t low, uint32_t high, uint32_t total_blocks) {

    uint32_t pivot = dbcsrMtx->bcolind[total_blocks + high]; 
    int32_t i = low - 1;
    uint32_t temp_ind;
    val_dt temp_val;

    for(uint32_t j = low; j <= high - 1; j++) {
        if(dbcsrMtx->bcolind[total_blocks + j] < pivot) {
            i++;
            // swap(i, j)
            temp_ind = dbcsrMtx->bcolind[total_blocks + i];
            dbcsrMtx->bcolind[total_blocks + i] = dbcsrMtx->bcolind[total_blocks + j];
            dbcsrMtx->bcolind[total_blocks + j] = temp_ind;
            for(uint32_t r=0; r<dbcsrMtx->row_block_size; r++) {
                for(uint32_t c=0; c<dbcsrMtx->col_block_size; c++) {
                    temp_val = dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + i * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c];
                    dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + i * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c] = dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + j * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c];
                    dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + j * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c] = temp_val;
                }
            }
        }
    }

    // swap(i+1, high)
    temp_ind = dbcsrMtx->bcolind[total_blocks + i+1];
    dbcsrMtx->bcolind[total_blocks + i+1] = dbcsrMtx->bcolind[total_blocks + high];
    dbcsrMtx->bcolind[total_blocks + high] = temp_ind;
    for(uint32_t r=0; r<dbcsrMtx->row_block_size; r++) {
        for(uint32_t c=0; c<dbcsrMtx->col_block_size; c++) {
            temp_val = dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + (i+1) * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c];
            dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + (i+1) * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c] = dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + high * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c];
            dbcsrMtx->bval[total_blocks * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + high * dbcsrMtx->row_block_size * dbcsrMtx->col_block_size + r * dbcsrMtx->col_block_size + c] = temp_val;
        }
    }
    return (i+1); 
}

/**
 * @brief quickSort for matrix in DBCSR format
 */
void quickSortDBCSRMatrix(struct DBCSRMatrix *dbcsrMtx, int32_t low, int32_t high, uint64_t total_blocks) {

    if (low < high) {
        uint32_t mid = partitionDBCSRMatrix(dbcsrMtx, low, high, total_blocks);

        quickSortDBCSRMatrix(dbcsrMtx, low, mid - 1, total_blocks); 
        quickSortDBCSRMatrix(dbcsrMtx, mid + 1, high, total_blocks); 
    }
}

/**
 * @brief Sort wrapper for matrix in DBCSR format
 */
void sortDBCSRMatrix(struct DBCSRMatrix *dbcsrMtx) {

    uint64_t total_blocks = 0;
    for (uint32_t r = 0; r < dbcsrMtx->horz_partitions; r++) {
        for (uint32_t c = 0; c < dbcsrMtx->vert_partitions; c++) {
            uint32_t partition = r * dbcsrMtx->vert_partitions + c;
            uint32_t ptr_offset = partition * (dbcsrMtx->num_block_rows + 1);

            for(uint32_t n=0; n<dbcsrMtx->num_block_rows; n++) {
                int32_t low = dbcsrMtx->browptr[ptr_offset + n]; 
                int32_t high = dbcsrMtx->browptr[ptr_offset + n+1] - 1; 

                quickSortDBCSRMatrix(dbcsrMtx, low, high, total_blocks);
            }
            total_blocks += dbcsrMtx->blocks_per_partition[partition];
        }
    }
}


/**
 * @brief deallocate matrix in DBCSR format 
 * @param matrix in DBCSR format
 */ 
void freeDBCSRMatrix(struct DBCSRMatrix *dbcsrMtx) {
    free(dbcsrMtx->nnzs_per_partition);
    free(dbcsrMtx->blocks_per_partition);
    free(dbcsrMtx->nnz_per_block);
    free(dbcsrMtx->browptr);
    free(dbcsrMtx->bcolind);
    free(dbcsrMtx->bval);
    free(dbcsrMtx);
}

/**
 * @brief convert matrix from DBCSR to DBCOO format 
 * @param matrix in DBCSR format
 */
struct DBCOOMatrix *dbcsr2dbcoo(struct DBCSRMatrix *dbcsrMtx) {
    struct DBCOOMatrix *dbcooMtx;
    dbcooMtx = (struct DBCOOMatrix *) malloc(sizeof(struct DBCOOMatrix));

    dbcooMtx->nrows = dbcsrMtx->nrows;
    dbcooMtx->ncols = dbcsrMtx->ncols;
    dbcooMtx->nnz = dbcsrMtx->nnz;
    dbcooMtx->nblocks = dbcsrMtx->nblocks;
    dbcooMtx->npartitions = dbcsrMtx->npartitions;
    dbcooMtx->horz_partitions = dbcsrMtx->horz_partitions;
    dbcooMtx->vert_partitions = dbcsrMtx->vert_partitions;
    dbcooMtx->tile_height = dbcsrMtx->tile_height;
    dbcooMtx->tile_width = dbcsrMtx->tile_width;
    dbcooMtx->num_block_rows = dbcsrMtx->num_block_rows;
    dbcooMtx->num_block_cols = dbcsrMtx->num_block_cols;
    dbcooMtx->num_rows_left = dbcsrMtx->num_rows_left;
    dbcooMtx->row_block_size = dbcsrMtx->row_block_size;
    dbcooMtx->col_block_size = dbcsrMtx->col_block_size;

    dbcooMtx->nnzs_per_partition = (uint32_t *) malloc(dbcooMtx->npartitions * sizeof(uint32_t));
    dbcooMtx->blocks_per_partition = (uint32_t *) malloc(dbcooMtx->npartitions * sizeof(uint32_t));
    dbcooMtx->bind = (struct bind_t *) malloc(dbcooMtx->nblocks * sizeof(struct bind_t));
    dbcooMtx->bval = (val_dt *) malloc((dbcooMtx->nblocks + 1) * dbcooMtx->row_block_size * dbcooMtx->col_block_size * sizeof(val_dt));
    dbcooMtx->nnz_per_block = (uint32_t *) malloc(dbcooMtx->nblocks * sizeof(uint32_t));

    uint64_t total_blocks = 0;
    for (uint32_t r = 0; r < dbcsrMtx->horz_partitions; r++) {
        for (uint32_t c = 0; c < dbcsrMtx->vert_partitions; c++) {
            uint32_t partition = r * dbcsrMtx->vert_partitions + c;
            uint32_t ptr_offset = partition * (dbcsrMtx->num_block_rows + 1);

            for(uint64_t n=0; n<dbcsrMtx->num_block_rows; n++) {
                for(uint64_t i=dbcsrMtx->browptr[ptr_offset + n]; i<dbcsrMtx->browptr[ptr_offset + n+1]; i++){
                    dbcooMtx->bind[total_blocks + i].rowind = n;
                    dbcooMtx->bind[total_blocks + i].colind = dbcsrMtx->bcolind[total_blocks + i];
                }
            }
            total_blocks += dbcsrMtx->blocks_per_partition[partition];
        }
    }

    memcpy(dbcooMtx->bval, dbcsrMtx->bval, (dbcooMtx->nblocks + 1) * dbcooMtx->row_block_size * dbcooMtx->col_block_size * sizeof(val_dt));
    memcpy(dbcooMtx->nnzs_per_partition, dbcsrMtx->nnzs_per_partition, dbcooMtx->npartitions * sizeof(uint32_t));
    memcpy(dbcooMtx->blocks_per_partition, dbcsrMtx->blocks_per_partition, dbcooMtx->npartitions * sizeof(uint32_t));
    memcpy(dbcooMtx->nnz_per_block, dbcsrMtx->nnz_per_block, dbcooMtx->nblocks * sizeof(uint32_t));

    return dbcooMtx;
}

/**
 * @brief deallocate matrix in DBCOO format 
 * @param matrix in DBCOO format
 */ 
void freeDBCOOMatrix(struct DBCOOMatrix *dbcooMtx) {
    free(dbcooMtx->nnzs_per_partition);
    free(dbcooMtx->blocks_per_partition);
    free(dbcooMtx->bind);
    free(dbcooMtx->bval);
    free(dbcooMtx->nnz_per_block);
    free(dbcooMtx);
}



#endif
