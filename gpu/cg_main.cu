#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <float.h>
#include <omp.h>
#include <limits>
#include <chrono>
#include <string>
#include <cassert>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include "mmio.h"
#include "sparse_helper.h"

using std::cout;
using std::endl;
using std::min;
using std::string;

#define CHECK_CUDA(func)                                               \
	{                                                                  \
		cudaError_t status = (func);                                   \
		if (status != cudaSuccess)                                     \
		{                                                              \
			printf("CUDA API failed at line %d with error: %s (%d)\n", \
				   __LINE__, cudaGetErrorString(status), status);      \
			return EXIT_FAILURE;                                       \
		}                                                              \
	}

#define CHECK_CUSPARSE(func)                                               \
	{                                                                      \
		cusparseStatus_t status = (func);                                  \
		if (status != CUSPARSE_STATUS_SUCCESS)                             \
		{                                                                  \
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
				   __LINE__, cusparseGetErrorString(status), status);      \
			return EXIT_FAILURE;                                           \
		}                                                                  \
	}

#define CHECK_CUBLAS(func)                                          \
	{                                                               \
		cublasStatus_t status = (func);                             \
		if (status != CUBLAS_STATUS_SUCCESS)                        \
		{                                                           \
			printf("CUBLAS API failed at line %d with error: %d\n", \
				   __LINE__, status);                               \
			return EXIT_FAILURE;                                    \
		}                                                           \
	}

struct GPUTimer
{
	GPUTimer()
	{
		cudaEventCreate(&start_);
		cudaEventCreate(&stop_);
		cudaEventRecord(start_, 0);
	}

	~GPUTimer()
	{
		cudaEventDestroy(start_);
		cudaEventDestroy(stop_);
	}

	void start()
	{
		cudaEventRecord(start_, 0);
	}

	float seconds()
	{
		cudaEventRecord(stop_, 0);
		cudaEventSynchronize(stop_);
		float time;
		cudaEventElapsedTime(&time, start_, stop_);
		return time * 1e-3;
	}

private:
	cudaEvent_t start_, stop_;
};

int CG_solver_gpu(
	int n,
	int nnz,
	int n_ite,
	double &process_time,
	int &n_ite_take,
	int *csrRowPtr,
	int *csrColIndex,
	double *csrVal,
	double *rhs,
	double *x0,
	bool p_f = false,
	double *x_final = NULL)
{

	GPUTimer timer;
	double p_time = 0.0;
	bool print_flag = (n_ite > 0) & p_f;
	process_time = 0.0;

	int *d_col, *d_row;
	double *d_val, *d_x;
	double *d_r, *d_p, *d_Ap;
	double r1;
	int k;
	double alpha, beta, neg_alpha;

	alpha = 1.0;
	neg_alpha = -1.0;
	beta = 0.0;

	double one_const = 1.0;
	double zero_const = 0.0;

	timer.start();

	CHECK_CUDA(cudaMalloc((void **)&d_col, nnz * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void **)&d_row, (n + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void **)&d_val, nnz * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_x, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_r, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_p, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_Ap, n * sizeof(double)));

	p_time = timer.seconds();
	if (print_flag)
	{
		printf("Device memory allocation time(s): %e\n", p_time);
	}

	timer.start();

	CHECK_CUDA(cudaMemcpy(d_col, csrColIndex, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_row, csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_val, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_p, x0, n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_x, x0, n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_r, rhs, n * sizeof(double), cudaMemcpyHostToDevice));

	p_time = timer.seconds();
	if (print_flag)
	{
		printf("Host to Device memory copy time(s): %e\n", p_time);
	}

	// CUSPARSE APIs
	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecP, vecAp;
	void *dBuffer = NULL;
	size_t bufferSize = 0;
	CHECK_CUSPARSE(cusparseCreate(&handle))
	// Create sparse matrix A in CSR format
	CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz,
									 d_row, d_col, d_val,
									 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
	// Create dense vector P
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F))
	// Create dense vector Ap
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F))
	// allocate an external buffer if needed
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecP, &beta, vecAp, CUDA_R_64F,
		CUSPARSE_MV_ALG_DEFAULT, &bufferSize))
	CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	CHECK_CUBLAS(cublasCreate(&cublasHandle))

	process_time = 0.0;

	if (print_flag)
	{
		printf("###GPU#GPU#GPU#GPU#GPU#GPU###\n");
	}

	timer.start();
	cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				 &one_const, matA, vecP, &zero_const, vecAp, CUDA_R_64F,
				 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

	cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1);

	cublasDcopy(cublasHandle, n, d_r, 1, d_p, 1);

	cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &r1);

	k = 0;

	if (print_flag)
	{
		printf("iteration = %3d, res = %e\n", k, r1);
	}

	if (n_ite < 0)
	{
		n_ite = 10;
	}

	double rr = r1;

	while (k < n_ite && rr >= 1.0e-8)
	{
		// A * P
		cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					 &one_const, matA, vecP, &zero_const, vecAp, CUDA_R_64F,
					 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

		// p * Ap
		double pAp = 0.0;
		cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &pAp);

		// alpha
		alpha = rr / pAp;
		neg_alpha = -alpha;

		// x = x + alpha * p
		cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_x, 1);

		// r = r - alpha * Ap
		cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1);

		double rr_old = rr;
		// dot(r_k+1)
		cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rr);

		beta = rr / rr_old;

		// p = r + beta * p
		cublasDscal(cublasHandle, n, &beta, d_p, 1);
		cublasDaxpy(cublasHandle, n, &one_const, d_r, 1, d_p, 1);

		k++;
		if (print_flag)
		{
			printf("iteration = %3d, res = %e\n", k, rr);
		}
	}

	p_time = timer.seconds();
	// printf("Compute time(s): %e\n", p_time);
	process_time = p_time;
	n_ite_take = k;

	cusparseDestroySpMat(matA);
	cusparseDestroyDnVec(vecP);
	cusparseDestroyDnVec(vecAp);
	cusparseDestroy(handle);

	cublasDestroy(cublasHandle);

	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ap);

	return 0;
}

int main(int argc, char *argv[])
{
	/* */
	bool p_f = false;

	if (argc < 3)
	{
		std::cout << "Usage: ./cg [sparse matrix A] [max ite] [print?]\n";
		return -1;
	}

	char *filename_A = argv[1];
	// char * filename_output = argv[2];
	// int N = 512;
	if (argc > 3)
	{
		p_f = true;
	}

	int M, K, nnz;
	vector<int> CSRRowPtr;
	vector<int> CSRColIndex;
	vector<double> CSRVal;

	cout << "Reading sparse A matrix ...";

	read_suitsparse_matrix_FP64(filename_A,
								CSRRowPtr,
								CSRColIndex,
								CSRVal,
								M,
								K,
								nnz,
								CSR);
	assert(M == K);

	cout << "Matrix size: \n";
	cout << "A: sparse matrix, " << M << " x " << K << ". NNZ = " << nnz << "\n";

	vector<double> b(M);
	cout << "Set b to [1,1,1...,1]^T \n";
	for (int i = 0; i < M; ++i)
	{
		b[i] = 1.0;
	}

	cout << "Set x0 to [0,0,0...,0]^T \n";
	vector<double> x0(M);
	for (int i = 0; i < M; ++i)
	{
		x0[i] = 0.0;
	}

	cout << "GPU Running\n";

	int n_ite = atoi(argv[2]);

	double process_time = 0.0;
	int n_ite_take = 0;
	CG_solver_gpu(M, nnz, 10,
				  process_time,
				  n_ite_take,
				  CSRRowPtr.data(), CSRColIndex.data(), CSRVal.data(),
				  b.data(), x0.data());

	process_time = 0.0;
	CG_solver_gpu(M, nnz, n_ite,
				  process_time,
				  n_ite_take,
				  CSRRowPtr.data(), CSRColIndex.data(), CSRVal.data(),
				  b.data(), x0.data(), p_f);

	cout << "GPU CG solver time(ms): " << process_time * 1000 << "\n";
	cout << "Iteration number: " << n_ite_take << "\n";

	return 0;
}
