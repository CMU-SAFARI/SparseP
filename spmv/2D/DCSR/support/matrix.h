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



#endif
