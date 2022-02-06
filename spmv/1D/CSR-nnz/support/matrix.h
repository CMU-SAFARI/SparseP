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
 * @brief read matrix from input fileName in COO format 
 * @param filename to read matrix (mtx format)
 */
static struct COOMatrix *readCOOMatrix(const char* fileName) {
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
            if((cooMtx->nrows % (8 / byte_dt)) != 0) { // Padding needed
                cooMtx->nrows += ((8 / byte_dt) - (cooMtx->nrows % (8 / byte_dt)));
            }
            if((cooMtx->ncols % (8 / byte_dt)) != 0) { // Padding needed
                cooMtx->ncols += ((8 / byte_dt) - (cooMtx->ncols % (8 / byte_dt)));
            }
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
            cooMtx->colind[i] = colindx - 1; // Convert indexes begin at 0
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
static void freeCOOMatrix(struct COOMatrix *cooMtx) {
    free(cooMtx->rowindx);
    free(cooMtx->colind);
    free(cooMtx->values);
    free(cooMtx);
}

/**
 * @brief convert matrix from COO to CSR 
 * @param matrix in COO format
 */
static struct CSRMatrix *coo2csr(struct COOMatrix *cooMtx) {

    struct CSRMatrix *csrMtx;
    csrMtx = (struct CSRMatrix *) malloc(sizeof(struct CSRMatrix));

    csrMtx->nrows = cooMtx->nrows;
    csrMtx->ncols = cooMtx->ncols;
    csrMtx->nnz = cooMtx->nnz;
    csrMtx->rowptr = (uint32_t *) calloc((csrMtx->nrows + 2), sizeof(uint32_t));
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
 * @brief deallocate matrix in CSR format 
 * @param matrix in CSR format
 */
static void freeCSRMatrix(struct CSRMatrix *csrMtx) {
    free(csrMtx->rowptr);
    free(csrMtx->colind);
    free(csrMtx->values);
    free(csrMtx);
}


#endif
