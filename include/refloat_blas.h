#ifndef REFLOAT_BLAS
#define REFLOAT_BLAS

void sparse_spmv_csr(int m,
					 int nnz,
					 double alpha,
					 int *csrRowPtr,
					 int *csrColIndex,
					 double *csrVal,
					 double *x,
					 double beta,
					 double *y,
					 int bsize);

void sparse_spmv_coo(int m,
					 int nnz,
					 double alpha,
					 int *cooRowIndex,
					 int *cooColIndex,
					 double *cooVal,
					 double *x,
					 double beta,
					 double *y,
					 int bsize);

void blas_copy(int n,
			   double *x,
			   int inc_x,
			   double *y,
			   int inc_y);

double blas_dot(int n,
				double *x,
				int inc_x,
				double *y,
				int inc_y);

void blas_axpby(int n,
				double alpha,
				double *x,
				int inc_x,
				double beta,
				double *y,
				int inc_y);

#endif