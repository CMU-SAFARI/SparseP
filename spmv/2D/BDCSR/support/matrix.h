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
    uint32_t *vert_tile_widths;
    uint32_t *nnzs_per_column;
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

    // Comput maximum width across all vertical tiles
    uint64_t total_nnzs = 0;
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
    free(bdcsrMtx->vert_tile_widths);
    free(bdcsrMtx->nnzs_per_column);
    free(bdcsrMtx->nnzs_per_vert_partition);
    free(bdcsrMtx->drowptr);
    free(bdcsrMtx->dcolind);
    free(bdcsrMtx->dval);
    free(bdcsrMtx);
}



#endif
