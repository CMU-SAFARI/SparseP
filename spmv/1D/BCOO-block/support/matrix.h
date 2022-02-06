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
    uint32_t *rowindx;
    uint32_t *colind;
    val_dt *values;
};

/**
 * @brief CSR matrix format 
 */
struct CSRMatrix {
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* rowptr;
    uint32_t* colind;
    val_dt *values;
};

/**
 * @brief BCSR matrix format 
 */
struct BCSRMatrix {
    uint64_t row_block_size;
    uint64_t col_block_size;
    uint32_t num_block_rows;
    uint32_t num_block_cols;
    uint64_t num_blocks;
    uint32_t num_rows_left;
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    uint32_t* browptr;  // row pointer
    uint32_t* bcolind;  // column indices
    val_dt* bval;       // nonzeros
    uint32_t* nnz_per_block;  // nnz per block
};

/**
 * @brief BCOO matrix format 
 */
struct BCOOMatrix {
    uint64_t row_block_size;
    uint64_t col_block_size;
    uint32_t num_block_rows;
    uint32_t num_block_cols;
    uint64_t num_blocks;
    uint32_t num_rows_left;
    uint32_t nrows;
    uint32_t ncols;
    uint32_t nnz;
    struct bind_t *bind; // indexes
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
    val_dt val;
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
            cooMtx->rowindx = (uint32_t *) calloc(cooMtx->nnz, sizeof(uint32_t));
            cooMtx->colind = (uint32_t *) calloc(cooMtx->nnz, sizeof(uint32_t));
            cooMtx->values = (val_dt *) calloc(cooMtx->nnz, sizeof(val_dt));
            done = true;
        } else {
            rowindx = atoi(token);
            token = strtok(NULL, " ");
            colindx = atoi(token);
            token = strtok(NULL, " ");
            val = (val_dt) (rand()%4 + 1);

            cooMtx->rowindx[i] = rowindx - 1; // Convert indexes to start at 0
            cooMtx->colind[i] = colindx - 1; // Convert indexes to start at 0
            cooMtx->values[i] = val; 
            i++;
        }
    }

    free(line);
    fclose(fp);
    return cooMtx;

}

/**
 * @brief deallocate matrix in COO format 
 * @param matrix in COO format
 */
void freeCOOMatrix(struct COOMatrix *cooMtx) {
    free(cooMtx->rowindx);
    free(cooMtx->colind);
    free(cooMtx->values);
    free(cooMtx);
}

/**
 * @brief convert matrix from COO to CSR format 
 * @param matrix in COO format
 */
struct CSRMatrix *coo2csr(struct COOMatrix *cooMtx) {

    struct CSRMatrix *csrMtx;
    csrMtx = (struct CSRMatrix *) malloc(sizeof(struct CSRMatrix));

    csrMtx->nrows = cooMtx->nrows;
    csrMtx->ncols = cooMtx->ncols;
    csrMtx->nnz = cooMtx->nnz;
    csrMtx->rowptr = (uint32_t *) calloc((csrMtx->nrows + 1), sizeof(uint32_t));
    csrMtx->colind = (uint32_t *) calloc((csrMtx->nnz + 1), sizeof(uint32_t));
    csrMtx->values = (val_dt *) calloc((csrMtx->nnz + 8), sizeof(val_dt)); // Padding needed

    for(unsigned int i = 0; i < cooMtx->nnz; ++i) {
        uint32_t rowIndx = cooMtx->rowindx[i];
        csrMtx->rowptr[rowIndx]++;
    }

    uint32_t sumBeforeNextRow = 0;
    for(unsigned int rowIndx = 0; rowIndx < csrMtx->nrows; ++rowIndx) {
        uint32_t sumBeforeRow = sumBeforeNextRow;
        sumBeforeNextRow += csrMtx->rowptr[rowIndx];
        csrMtx->rowptr[rowIndx] = sumBeforeRow;
    }
    csrMtx->rowptr[csrMtx->nrows] = sumBeforeNextRow;

    for(unsigned int i = 0; i < cooMtx->nnz; ++i) {
        uint32_t rowIndx = cooMtx->rowindx[i];
        uint32_t nnzIndx = csrMtx->rowptr[rowIndx]++;
        csrMtx->colind[nnzIndx] = cooMtx->colind[i];
        csrMtx->values[nnzIndx] = cooMtx->values[i];
    }

    for(unsigned int rowIndx = csrMtx->nrows - 1; rowIndx > 0; --rowIndx) {
        csrMtx->rowptr[rowIndx] = csrMtx->rowptr[rowIndx - 1];
    }
    csrMtx->rowptr[0] = 0;

    return csrMtx;

}

/**
 * @brief print matrix in CSR format
 */
void printCSRMatrix(struct CSRMatrix *A) {

    for(unsigned int rowIndx = 0; rowIndx < A->nrows; ++rowIndx) {
        printf("Row: %d\n", rowIndx);
        for(unsigned int i = A->rowptr[rowIndx]; i < A->rowptr[rowIndx + 1]; ++i) {
            unsigned int colIndx = A->colind[i];
            val_dt value = A->values[i];
            printf("(%d, %d) ", colIndx, value);
        }
        printf("\n");
    }
}

/**
 * @brief deallocate matrix in CSR format 
 * @param matrix in CSR format
 */
void freeCSRMatrix(struct CSRMatrix *csrMtx) {
    free(csrMtx->rowptr);
    free(csrMtx->colind);
    free(csrMtx->values);
    free(csrMtx);
}

/**
 * @brief convert matrix from CSR to BCSR format 
 * @param matrix in CSR format
 * Taken from OSKI: A library of automatically tuned sparse matrix kernels
 * http://bebop.cs.berkeley.edu/oski/downloads.html
 */
struct BCSRMatrix *csr2bcsr(struct CSRMatrix *csrMtx, uint32_t row_block_size, uint32_t col_block_size) {
    struct BCSRMatrix *bcsrMtx;
    bcsrMtx = (struct BCSRMatrix *) malloc(sizeof(struct BCSRMatrix));

    uint32_t num_block_rows = (csrMtx->nrows + row_block_size - 1) / row_block_size;
    uint32_t num_block_cols = (csrMtx->ncols + col_block_size - 1) / col_block_size;
    uint32_t num_rows_left = csrMtx->nrows % row_block_size;

    bcsrMtx->nrows = csrMtx->nrows;
    bcsrMtx->ncols = csrMtx->ncols;
    bcsrMtx->nnz = csrMtx->nnz;
    bcsrMtx->row_block_size = row_block_size; 
    bcsrMtx->col_block_size = col_block_size;
    bcsrMtx->num_block_rows = num_block_rows;
    bcsrMtx->num_block_cols = num_block_cols;
    bcsrMtx->num_rows_left = num_rows_left;

    uint32_t *bAp;
    uint32_t *bAj;
    val_dt *bAx;

    //tmp variables
    uint32_t num_blocks = 0;
    uint32_t *block_count;

    uint32_t *bAp_next;
    uint64_t I, J;
    uint64_t i, j, k, j0, di;
    val_dt a_ij;

    bAp = (uint32_t *) malloc((num_block_rows+2) * sizeof(uint32_t));
    bAp_next = (uint32_t *) malloc((num_block_rows+2) * sizeof(uint32_t));
    block_count = (uint32_t *) malloc((num_block_cols) * sizeof(uint32_t));
    memset(block_count, 0, num_block_cols * sizeof(uint32_t));

    //Phase I: Count the exact number of new blocks to create.
    bAp[0] = 0; 
    if(num_rows_left == 0) {    
        for(I=0; I<num_block_rows; I++) {    
            for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {    
                for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {    
                    j = csrMtx->colind[k];
                    J = j/col_block_size;
                    if(block_count[J] == 0) {    
                        num_blocks++;
                        block_count[J]++;
                    }
                }
            }
            bAp[I+1] = num_blocks;
            for(i = 0; i < num_block_cols; i++)
                block_count[i] = 0;
        }
    } else {
        for(I=0; I<num_block_rows-1; I++) {
            for(i=I * row_block_size; i < (I+1) * row_block_size; i++) {
                for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                    j = csrMtx->colind[k];
                    J = j/col_block_size;
                    if(block_count[J] == 0) {
                        num_blocks++;
                        block_count[J]++;
                    }
                }
            }
            bAp[I+1] = num_blocks;
            for(i = 0; i < num_block_cols; i++)
                block_count[i] = 0;
        }
        for(i = (num_block_rows-1) * row_block_size; i < ((num_block_rows-1) * row_block_size + num_rows_left); i++) {
            for (k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                j = csrMtx->colind[k];
                J = j/col_block_size;
                if (block_count[J] == 0) {
                    num_blocks++;
                    block_count[J]++;
                }
            }
        }
        bAp[num_block_rows] = num_blocks;
        for(i = 0; i < num_block_cols; i++)
            block_count[i] = 0;
    }

    bcsrMtx->num_blocks = num_blocks;
    bAj = (uint32_t *) malloc((num_blocks + 1) * sizeof(uint32_t));
    bAx = (val_dt *) calloc(((num_blocks + 1) * row_block_size * col_block_size), sizeof(val_dt));
    val_dt *blocks = (val_dt *) malloc((row_block_size * col_block_size * num_block_cols) * sizeof(val_dt));
    memset(blocks, 0, row_block_size * col_block_size * num_block_cols * sizeof(val_dt));
    memcpy(bAp_next, bAp, (num_block_rows+1) * sizeof(uint32_t));
    bcsrMtx->nnz_per_block = (uint32_t *) calloc(num_blocks, sizeof(uint32_t));


    //Phase II: Copy all blocks.
    if(num_rows_left == 0) {
        for(I=0; I < num_block_rows; I++) {
            for(i = I * row_block_size, di=0; di<row_block_size; di++, i++) {
                for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                    j = csrMtx->colind[k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;
                    a_ij = csrMtx->values[k];
                    blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                    block_count[J]++;
                }
            }
            for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                    j = csrMtx->colind[k];
                    J = j / col_block_size;
                    j0 = J * col_block_size; 

                    if(block_count[J] > 0) {
                        uint64_t k_next = bAp_next[I]; 
                        bAj[k_next] = J; 
                        memcpy(bAx + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                        bAp_next[I]++;
                        assert(bAp_next[I] <= bAp[I+1]);
                        block_count[J] = 0;
                        memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                    }
                }
            }
        }
    } else {
        for(I = 0; I < num_block_rows-1; I++) {
            for(i = I*row_block_size, di=0; di<row_block_size; di++, i++) {
                for(k = csrMtx->rowptr[i]; k<csrMtx->rowptr[i+1]; k++) {
                    j = csrMtx->colind[k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;
                    a_ij = csrMtx->values[k];
                    blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                    block_count[J]++;
                }
            }

            for(i = I*row_block_size, di=0; di < row_block_size; di++, i++) {
                for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                    j = csrMtx->colind[k];
                    J = j / col_block_size;
                    j0 = J * col_block_size;

                    if (block_count[J] > 0) {
                        uint64_t k_next = bAp_next[I];
                        bAj[k_next] = J;
                        memcpy(bAx + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                        bAp_next[I]++;
                        assert(bAp_next[I] <= bAp[I+1]);
                        block_count[J] = 0;
                        memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                    }
                }
            }

        }

        for(i = (num_block_rows-1)*row_block_size, di=0; di < num_rows_left; di++, i++) {
            for(k = csrMtx->rowptr[i]; k < csrMtx->rowptr[i+1]; k++) {
                j = csrMtx->colind[k];
                J = j / col_block_size;
                j0 = J * col_block_size;
                a_ij = csrMtx->values[k];
                blocks[J * row_block_size * col_block_size + di * col_block_size + j - j0] = a_ij;
                block_count[J]++;
            }
        }

        for(i = (num_block_rows-1)*row_block_size, di=0; di<num_rows_left; di++, i++) {
            for(k = csrMtx->rowptr[i]; k<csrMtx->rowptr[i+1]; k++) {
                j = csrMtx->colind[k];
                J = j / col_block_size;
                j0 = J * col_block_size;

                if(block_count[J] > 0) {
                    uint64_t k_next = bAp_next[num_block_rows-1];
                    bAj[k_next] = J; 
                    memcpy(bAx + k_next * row_block_size * col_block_size, blocks + J * row_block_size * col_block_size, row_block_size * col_block_size * sizeof(val_dt));
                    bAp_next[num_block_rows-1]++;
                    assert(bAp_next[num_block_rows-1] <= bAp[num_block_rows]);
                    block_count[J] = 0;
                    memset(blocks + J * col_block_size * row_block_size, 0, row_block_size * col_block_size * sizeof(val_dt));
                }
            }
        }

    }


    free(block_count);
    free(blocks);
    free(bAp_next);

    bcsrMtx->browptr = bAp;
    bcsrMtx->bcolind = bAj;
    bcsrMtx->bval = bAx;

    return bcsrMtx;
}

/**
 * @brief partition function for quickSort 
 */
uint32_t partitionBCSRMatrix(struct BCSRMatrix *bcsrMtx, uint32_t low, uint32_t high) {

    uint32_t pivot = bcsrMtx->bcolind[high]; 
    uint32_t i = low - 1;
    uint32_t temp_ind;
    val_dt temp_val;

    for(uint64_t j = low; j <= high - 1; j++) {
        if(bcsrMtx->bcolind[j] < pivot) {
            i++;
            // swap(i, j)
            temp_ind = bcsrMtx->bcolind[i];
            bcsrMtx->bcolind[i] = bcsrMtx->bcolind[j];
            bcsrMtx->bcolind[j] = temp_ind;
            for(uint32_t r=0; r<bcsrMtx->row_block_size; r++) {
                for(uint32_t c=0; c<bcsrMtx->col_block_size; c++) {
                    temp_val = bcsrMtx->bval[i * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c];
                    bcsrMtx->bval[i * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c] = bcsrMtx->bval[j * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c];
                    bcsrMtx->bval[j * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c] = temp_val;
                }
            }
        }
    }

    // swap(i+1, high)
    temp_ind = bcsrMtx->bcolind[i+1];
    bcsrMtx->bcolind[i+1] = bcsrMtx->bcolind[high];
    bcsrMtx->bcolind[high] = temp_ind;
    for(uint32_t r=0; r<bcsrMtx->row_block_size; r++) {
        for(uint32_t c=0; c<bcsrMtx->col_block_size; c++) {
            temp_val = bcsrMtx->bval[(i+1) * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c];
            bcsrMtx->bval[(i+1) * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c] = bcsrMtx->bval[high * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c];
            bcsrMtx->bval[high * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c] = temp_val;
        }
    }
    return (i+1); 
}

/**
 * @brief quickSort for matrix in BCSR format
 */
void quickSortBCSRMatrix(struct BCSRMatrix *bcsrMtx, int32_t low, int32_t high) {

    if (low < high) {
        uint32_t mid = partitionBCSRMatrix(bcsrMtx, low, high);

        quickSortBCSRMatrix(bcsrMtx, low, mid - 1); 
        quickSortBCSRMatrix(bcsrMtx, mid + 1, high); 
    }
}

/**
 * @brief Sort wrapper for matrix in BCSR format
 */
void sortBCSRMatrix(struct BCSRMatrix *bcsrMtx) {

    for(uint32_t n=0; n<bcsrMtx->num_block_rows; n++) {
        int32_t low = bcsrMtx->browptr[n]; 
        int32_t high = bcsrMtx->browptr[n+1] - 1; 

        quickSortBCSRMatrix(bcsrMtx, low, high);
    }

}

/**
 * @brief count nnz per block in BCSR format matrix
 */
void countNNZperBlockBCSRMatrix(struct BCSRMatrix *bcsrMtx) {
    double occupancy = 0;
    for(uint32_t n=0; n<bcsrMtx->num_block_rows; n++) {
        for(uint32_t i=bcsrMtx->browptr[n]; i<bcsrMtx->browptr[n+1]; i++){
            uint32_t j = bcsrMtx->bcolind[i];
            for(uint32_t r=0; r<bcsrMtx->row_block_size; r++){
                for(uint32_t c=0; c<bcsrMtx->col_block_size; c++) {
                    if (bcsrMtx->bval[i * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c] != 0) 
                        bcsrMtx->nnz_per_block[i]++;
                }
            }
            occupancy += ((double) bcsrMtx->nnz_per_block[i] / (double) (bcsrMtx->row_block_size * bcsrMtx->col_block_size));
        }
    }
    //printf("Occupancy ratio %lf\n", occupancy / bcsrMtx->num_blocks);
}

/**
 * @brief print matrix in BCSR format
 */
void printBCSRMatrix(struct BCSRMatrix *bcsrMtx) {

    printf("Matrix of size %d x %d NNZ %d\n", bcsrMtx->nrows, bcsrMtx->ncols, bcsrMtx->nnz);
    printf("Total Blocks: %ld\n", bcsrMtx->num_blocks);
    printf("Num Block Rows: %d, Num Block Cols: %d of block %ld x %ld\n", bcsrMtx->num_block_rows, bcsrMtx->num_block_cols, bcsrMtx->row_block_size, bcsrMtx->col_block_size);

    for(uint64_t n=0; n<bcsrMtx->num_block_rows; n++) {
        for(uint64_t i=bcsrMtx->browptr[n]; i<bcsrMtx->browptr[n+1]; i++){
            uint64_t j = bcsrMtx->bcolind[i];
            printf("Block (%ld, %ld) NNZ: %d\n", n, j, bcsrMtx->nnz_per_block[i]);
            for(uint64_t r=0; r<bcsrMtx->row_block_size; r++){
                for(uint64_t c=0; c<bcsrMtx->col_block_size; c++) {
                    printf("%d ", bcsrMtx->bval[i * bcsrMtx->row_block_size * bcsrMtx->col_block_size + r * bcsrMtx->col_block_size + c]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

/**
 * @brief deallocate matrix in BCSR format 
 * @param matrix in BCSR format
 */
void freeBCSRMatrix(struct BCSRMatrix *bcsrMtx) {
    free(bcsrMtx->browptr);
    free(bcsrMtx->bcolind);
    free(bcsrMtx->bval);
    free(bcsrMtx->nnz_per_block);
    free(bcsrMtx);
}


/**
 * @brief convert matrix from BCSR to BCOO format 
 * @param matrix in BCSR format
 */
struct BCOOMatrix *bcsr2bcoo(struct BCSRMatrix *bcsrMtx) {
    struct BCOOMatrix *bcooMtx;
    bcooMtx = (struct BCOOMatrix *) malloc(sizeof(struct BCOOMatrix));

    bcooMtx->nrows = bcsrMtx->nrows;
    bcooMtx->ncols = bcsrMtx->ncols;
    bcooMtx->nnz = bcsrMtx->nnz;
    bcooMtx->num_block_rows = bcsrMtx->num_block_rows;
    bcooMtx->num_block_cols = bcsrMtx->num_block_cols;
    bcooMtx->num_blocks = bcsrMtx->num_blocks;
    bcooMtx->num_rows_left = bcsrMtx->num_rows_left;
    bcooMtx->row_block_size = bcsrMtx->row_block_size;
    bcooMtx->col_block_size = bcsrMtx->col_block_size;

    bcooMtx->bind = (struct bind_t *) malloc(bcooMtx->num_blocks * sizeof(struct bind_t));
    bcooMtx->bval = (val_dt *) malloc((bcooMtx->num_blocks + 1) * bcooMtx->row_block_size * bcooMtx->col_block_size * sizeof(val_dt));
    bcooMtx->nnz_per_block = (uint32_t *) malloc(bcooMtx->num_blocks * sizeof(uint32_t));


    for(uint64_t n=0; n<bcsrMtx->num_block_rows; n++) {
        for(uint64_t i=bcsrMtx->browptr[n]; i<bcsrMtx->browptr[n+1]; i++){
            bcooMtx->bind[i].rowind = n;
            bcooMtx->bind[i].colind = bcsrMtx->bcolind[i];
        }
    }

    memcpy(bcooMtx->bval, bcsrMtx->bval, (bcooMtx->num_blocks + 1) * bcooMtx->row_block_size * bcooMtx->col_block_size * sizeof(val_dt));
    memcpy(bcooMtx->nnz_per_block, bcsrMtx->nnz_per_block, bcooMtx->num_blocks * sizeof(uint32_t));
    return bcooMtx;
}

/**
 * @brief print matrix in BCOO format
 */
void printBCOOMatrix(struct BCOOMatrix *bcooMtx) {

    printf("Matrix of size %d x %d NNZ %d\n", bcooMtx->nrows, bcooMtx->ncols, bcooMtx->nnz);
    printf("Total Blocks: %ld\n", bcooMtx->num_blocks);
    printf("Num Block Rows: %d, Num Block Cols: %d of block %ld x %ld\n", bcooMtx->num_block_rows, bcooMtx->num_block_cols, bcooMtx->row_block_size, bcooMtx->col_block_size);

    for(uint64_t n=0; n<bcooMtx->num_blocks; n++) {
        uint32_t i = bcooMtx->bind[n].rowind;
        uint32_t j = bcooMtx->bind[n].colind;
        printf("Block (%d, %d) NNZ: %d\n", i, j, bcooMtx->nnz_per_block[n]);
        for(uint64_t r=0; r<bcooMtx->row_block_size; r++){
            for(uint64_t c=0; c<bcooMtx->col_block_size; c++) {
                printf("%d ", bcooMtx->bval[i * bcooMtx->row_block_size * bcooMtx->col_block_size + r * bcooMtx->col_block_size + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

/**
 * @brief deallocate matrix in BCOO format 
 * @param matrix in BCOO format
 */
void freeBCOOMatrix(struct BCOOMatrix *bcooMtx) {
    free(bcooMtx->bind);
    free(bcooMtx->bval);
    free(bcooMtx->nnz_per_block);
    free(bcooMtx);
}


#endif
