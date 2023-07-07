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

int BICG_solver_gpu(
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
	double *d_r0_bar, *d_v, *d_s, *d_t;
	double *d_Ax, *d_h;
	double r1;
	int k;
	double alpha, beta, r0, dot;

	alpha = 1.0;
	const double neg_alpha = -1.0;
	beta = 0.0;
	r0 = 0.0;

	double alpha_h = 1.0;
	double beta_h = 0.0;
	double omega_h = 1.0;
	double rho_h = 1.0;
	double rho_new_h = 1.0;
	double res_h = 0.0;

	const double one_const = 1.0;
	const double zero_const = 0.0;
	const vector<double> vec_zero(n, 0.0);

	timer.start();

	CHECK_CUDA(cudaMalloc((void **)&d_col, nnz * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void **)&d_row, (n + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void **)&d_val, nnz * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_x, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_r, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_p, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_Ax, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_Ap, n * sizeof(double)));

	CHECK_CUDA(cudaMalloc((void **)&d_v, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_s, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_t, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_h, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void **)&d_r0_bar, n * sizeof(double)));

	p_time = timer.seconds();
	if (print_flag)
	{
		printf("Device memory allocation time(s): %e\n", p_time);
	}

	timer.start();

	CHECK_CUDA(cudaMemcpy(d_col, csrColIndex, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_row, csrRowPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_val, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_x, x0, n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_r, rhs, n * sizeof(double), cudaMemcpyHostToDevice));

	CHECK_CUDA(cudaMemcpy(d_v, vec_zero.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_p, vec_zero.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_s, vec_zero.data(), n * sizeof(double), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_t, vec_zero.data(), n * sizeof(double), cudaMemcpyHostToDevice));

	p_time = timer.seconds();
	if (print_flag)
	{
		printf("Host to Device memory copy time(s): %e\n", p_time);
	}

	// CUSPARSE APIs
	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecP, vecX, vecS, vecAx, vecV, vecT, vecAp;
	void *dBuffer = NULL;
	size_t bufferSize = 0;
	CHECK_CUSPARSE(cusparseCreate(&handle))
	// Create sparse matrix A in CSR format
	CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz,
									 d_row, d_col, d_val,
									 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
									 CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
	// Create dense vectors
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecS, n, d_s, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecAx, n, d_Ax, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecV, n, d_v, CUDA_R_64F))
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecT, n, d_t, CUDA_R_64F))

	// allocate an external buffer if needed
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecP, &beta, vecAx, CUDA_R_64F,
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
				 &one_const, matA, vecX, &zero_const, vecAx, CUDA_R_64F,
				 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

	cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ax, 1, d_r, 1);
	cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &r1);

	// r0 hat
	cublasDcopy(cublasHandle, n, d_r, 1, d_r0_bar, 1);

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
		// rho_new = r0_bar' * r;
		cublasDdot(cublasHandle, n, d_r0_bar, 1, d_r, 1, &rho_new_h);

		beta_h = (rho_new_h / rho_h) * (alpha_h / omega_h);

		// d_p = d_r + beta_h * (d_p - omega_h * d_v)
		alpha = 0.0 - omega_h;
		cublasDaxpy(cublasHandle, n, &alpha, d_v, 1, d_p, 1); // d_p <- (- omega) * d_v + d_p
		cublasDscal(cublasHandle, n, &beta_h, d_p, 1);		  // d_p <- d_p * beta_h
		alpha = 1.0;
		cublasDaxpy(cublasHandle, n, &alpha, d_r, 1, d_p, 1); // d_p <- 1.0 * d_r + d_p

		// d_v = A * d_p;
		cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					 &one_const, matA, vecP, &zero_const, vecV, CUDA_R_64F,
					 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

		// alpha_h = rho_new_h / (d_r0_bar' * d_v);
		cublasDdot(cublasHandle, n, d_r0_bar, 1, d_v, 1, &dot);
		alpha_h = rho_new_h / dot;

		// d_h = d_x + alpha_h * d_p;
		cublasDcopy(cublasHandle, n, d_x, 1, d_h, 1); // d_h <- d_x
		alpha = alpha_h;
		cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_h, 1); // d_h <- (alpha_h) * d_p + d_h

		// d_s = d_r - alpha_h * d_v;
		cublasDcopy(cublasHandle, n, d_r, 1, d_s, 1); // d_s <- d_r
		alpha = 0.0 - alpha_h;
		cublasDaxpy(cublasHandle, n, &alpha, d_v, 1, d_s, 1); // d_s <- (- alpha_h) * d_v + d_s

		// d_t = A * d_s;
		cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					 &one_const, matA, vecS, &zero_const, vecT, CUDA_R_64F,
					 CUSPARSE_MV_ALG_DEFAULT, dBuffer);

		// omega_h = (d_t' * d_s) / (d_t' * d_t);
		cublasDdot(cublasHandle, n, d_t, 1, d_s, 1, &omega_h);
		cublasDdot(cublasHandle, n, d_t, 1, d_t, 1, &dot);
		omega_h = omega_h / dot;

		// d_x = d_h + omega_h * d_s;
		cublasDcopy(cublasHandle, n, d_h, 1, d_x, 1); // d_x <- d_h
		alpha = omega_h;
		cublasDaxpy(cublasHandle, n, &alpha, d_s, 1, d_x, 1); // d_x <- (omega_h) * d_s + d_x

		// d_r = d_s - omega_h * d_t;
		cublasDcopy(cublasHandle, n, d_s, 1, d_r, 1); // d_r <- d_s
		alpha = 0.0 - omega_h;
		cublasDaxpy(cublasHandle, n, &alpha, d_t, 1, d_r, 1); // d_r <- (- omega_h) * d_t + d_r

		rho_h = rho_new_h;

		cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rr);

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
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecS);
	cusparseDestroyDnVec(vecAx);
	cusparseDestroyDnVec(vecV);
	cusparseDestroyDnVec(vecT);
	cusparseDestroy(handle);
	cublasDestroy(cublasHandle);

	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);
	cudaFree(d_Ap);

	cudaFree(d_v);
	cudaFree(d_s);
	cudaFree(d_t);
	cudaFree(d_h);
	cudaFree(d_r0_bar);

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
	BICG_solver_gpu(M, nnz, 10,
					process_time,
					n_ite_take,
					CSRRowPtr.data(), CSRColIndex.data(), CSRVal.data(),
					b.data(), x0.data());

	process_time = 0.0;
	BICG_solver_gpu(M, nnz, n_ite,
					process_time,
					n_ite_take,
					CSRRowPtr.data(), CSRColIndex.data(), CSRVal.data(),
					b.data(), x0.data(), p_f);

	cout << "GPU BiCG solver time(s): " << process_time << "\n";
	cout << "Iteration number: " << n_ite_take << "\n";

	return 0;
}