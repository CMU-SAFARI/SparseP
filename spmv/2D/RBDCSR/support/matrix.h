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


#endif
