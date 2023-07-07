#include <stdio.h>
#include <stdlib.h>

void sparse_spmv_csr(int m,
                     int nnz,
                     double alpha,
                     int * csrRowPtr,
                     int * csrColIndex,
                     double * csrVal,
                     double * x,
                     double beta,
                     double * y,
                     int bsize) {
    for(int i = 0; i < m; ++i) {
        double psum = 0.0;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; ++j) {
            psum += csrVal[j] * x[csrColIndex[j]];
        }
        y[i] = alpha * psum + beta * y[i];
    }
}


void sparse_spmv_coo(int m,
                     int nnz,
                     double alpha,
                     int * cooRowIndex,
                     int * cooColIndex,
                     double * cooVal,
                     double * x,
                     double beta,
                     double * y,
                     int bsize) {
    for (int i = 0; i < m; ++i) {
        y[i] *= beta;
    }
    int pre_row = cooRowIndex[0];
    int pre_col = cooColIndex[0];
    double psum = 0.0;
    for(int i = 0; i < nnz; ++i) {
        if (cooRowIndex[i] != pre_row ||
            ((pre_col / bsize) != (cooColIndex[i] / bsize))) {
            y[pre_row] += alpha * psum;
            psum = 0.0;
            pre_row = cooRowIndex[i];
            pre_col = cooColIndex[i];
        }
        psum += cooVal[i] * x[cooColIndex[i]];
    }
    y[pre_row] += alpha * psum;
}


void blas_copy(int n,
               double * x,
               int inc_x,
               double * y,
               int inc_y) {
    for (int i = 0; i < n; ++i) {
        y[i*inc_y] = x[i*inc_x];
    }
}

double blas_dot(int n,
                double * x,
                int inc_x,
                double * y,
                int inc_y) {
    double ans = 0.0;
    for (int i = 0; i < n; ++i) {
        ans += x[i*inc_x] * y[i*inc_y];
    }
    return ans;
}

void blas_axpby(int n,
                double alpha,
                double * x,
                int inc_x,
                double beta,
                double * y,
                int inc_y) {
    for(int i = 0; i < n; ++i) {
        y[i*inc_y] = alpha * x[i*inc_x] + beta * y[i*inc_y];
    }
}