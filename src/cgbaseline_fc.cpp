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
#include <cassert>

#include "mmio.h"
#include "sparse_helper.h"
#include "refloat_helper.h"
#include "refloat_format.h"
#include "refloat_blas.h"
#include "refloat_solver.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
    const int bsize = 128;

    int alg = 0;

    if (argc < 3) {
        std::cout << "Usage: ./cgbaseline_fc [sparse matrix A] [number of interations(s)] [alg: 0 -> cg, 1 -> bicg]\n";
        return -1;
    }

    char *filename_A = argv[1];
    int n_ite = atoi(argv[2]);

    if (argc > 3) {
        alg = atoi(argv[3]);
        if ((alg < 0) || (alg > 1)) {
            cout << "Unknow solver algorithm./n";
            return -1;
        }
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

    vector<double> b(M);
    vector<double> x0(M);

    for (int i = 0; i < M; ++i) {
        b[i] = 1.0;
        x0[i] = 0.0;
    }
    cout << "Set b to [1,1,1...,1]^T \n";
    cout << "Set x0 to [0,0,0...,0]^T \n";

    int solver_ite = 0;
    double process_time = 0.0;

    vector<int> bcooRowIndex(nnz);
	vector<int> bcooColIndex(nnz);
	vector<double> bcooVal(nnz);
	int num_blocks = 0;

    csr2bcoo_refloat(M, 
                     M, 
                     nnz, 
			         bsize, 
			         CSRRowPtr.data(), 
                     CSRColIndex.data(),
                     CSRVal.data(),
			         bcooRowIndex.data(), 
			         bcooColIndex.data(), 
			         bcooVal.data(), 
			         num_blocks,
			         6,
			         52);

    solverinfo s_info;
    s_info.platfrom = BaselineAcc;
    s_info.bsize = bsize;
    s_info.num_blocks = num_blocks;
    s_info.matrix_exponent = 6;
    s_info.matrix_fraction = 52;
    s_info.vec_exponent = 6;
    s_info.vec_fraction = 52;
    s_info.th = 1.0e-8;
    s_info.sparse_spmv = sparse_spmv_csr;

    if (alg == 0) {
        cout << "@@@ Baseline Acc (forced function correct) CG Solver...\n";
        cout << "@@@ Matrix: (6 + 52)\n";
        cout << "@@@ Vector: (6 + 52)\n";
        solver_ite =
            CG_solver(M,
                      nnz,
                      n_ite,
                      process_time,
                      CSRRowPtr.data(),
                      CSRColIndex.data(),
                      CSRVal.data(),
                      b.data(),
                      x0.data(),
                      s_info);
    } else if (alg == 1) {
        cout << "@@@ Baseline Acc (forced function correct) BiCG Solver...\n";
        cout << "@@@ Matrix: (6 + 52)\n";
        cout << "@@@ Vector: (6 + 52)\n";
        solver_ite =
            BiCG_solver(M,
                        nnz,
                        n_ite,
                        process_time,
                        CSRRowPtr.data(),
                        CSRColIndex.data(),
                        CSRVal.data(),
                        b.data(),
                        x0.data(),
                        s_info);
    }
    
    if (solver_ite == -1) {
        cout << "Solver abnomal exit!\n";
    } else {
        cout << "Total iterations: " << solver_ite << endl;
        cout << "Total processing time(ms): " << process_time * 1000 << endl << endl;
    }
    
    return 0;
}