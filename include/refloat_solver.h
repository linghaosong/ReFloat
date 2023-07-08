#ifndef REFLOAT_SOLVER
#define REFLOAT_SOLVER

enum platform_t
{
	CPU,
	BaselineAcc,
	ReFloatAcc
};

struct solverinfo
{
	platform_t platfrom = CPU;
	int bsize = 0;
	int matrix_exponent = 11;
	int matrix_fraction = 52;
	int vec_exponent = 11;
	int vec_fraction = 52;
	int num_blocks = 0;
	double th = 1.0e-8;
	void (*sparse_spmv)(int, int, double, int *, int *, double *, double *, double, double *, int) = NULL;
	void (*vector2refloat)(const int, const int, double *, const int, const int) = NULL;
};

struct reraminfo
{
	const int num_aval_xbar = 128 * 128 * 64;
	const double t_write_per_row = 50.88e-9;
	const double t_comp = 107.0e-9;
	const double freq = 1.5e+9;
};

int BiCG_solver(int n,
				int nnz,
				int n_ite,
				double &process_time,
				int *csrRowPtr,
				int *csrColIndex,
				double *csrVal,
				double *rhs,
				double *x0,
				solverinfo &s_info);

int CG_solver(int n,
			  int nnz,
			  int n_ite,
			  double &process_time,
			  int *csrRowPtr,
			  int *csrColIndex,
			  double *csrVal,
			  double *rhs,
			  double *x0,
			  solverinfo &s_info);

int CG_HMP_solver(int n,
				  int nnz,
				  int n_ite,
				  double &process_time,
				  int *csrRowPtr,
				  int *csrColIndex,
				  double *csrVal,
				  double *rhs,
				  double *x0,
				  solverinfo & s_info);

int BiCG_HMP_solver(int n,
					int nnz,
					int n_ite,
					double &process_time,
					int *csrRowPtr,
					int *csrColIndex,
					double *csrVal,
					double *rhs,
					double *x0,
					solverinfo & s_info);

#endif