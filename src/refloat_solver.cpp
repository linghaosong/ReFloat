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

#include "refloat_helper.h"
#include "refloat_blas.h"
#include "refloat_solver.h"

using std::cout;
using std::endl;
using std::min;
using std::isinf;
using std::isnan;

int BiCG_solver(int n,
                int nnz,
                int n_ite,
                double &process_time,
                int *csrRowPtr,
                int *csrColIndex,
                double *csrVal,
                double *rhs,
                double *x0,
                solverinfo & s_info) {
    double curr_time = get_time();

    void (*sparse_spmv)(int, int, double, int *, int *, double *, double *, double, double *, int) = s_info.sparse_spmv;
    const int platform = s_info.platfrom;
    const int bsize = s_info.bsize;
    void (*vector2refloat)(const int, const int, double *, const int exponent, const int) = s_info.vector2refloat;
    const int vec_exponent = s_info.vec_exponent;
    const int vec_fraction = s_info.vec_fraction;
    const double th = s_info.th;

    int k;
    double alpha = 1.0;
    double beta = 0.0;
    double omega = 1.0;
    double rho = 1.0;
    double rho_new = 1.0;
    double res = 0.0;

    double *r = new double[n];
    double *r0_bar = new double[n];
    double *v = new double[n];
    double *p = new double[n];
    double *s = new double[n];
    double *t = new double[n];

    for (int i = 0; i < n; ++i) {
        r[i] = 0.0;
        v[i] = 0.0;
        p[i] = 0.0;
        s[i] = 0.0;
        t[i] = 0.0;
    }

    const reraminfo r_cfg;
    double t_sim = 0.0;
    const int max_num_engines = (s_info.platfrom == CPU)? 1 : 
                                    (r_cfg.num_aval_xbar / 
                                    (4 * ((1 << s_info.matrix_exponent) + s_info.matrix_fraction + 1)));
    const int xbar_cycle = (1 << s_info.matrix_exponent) + s_info.matrix_fraction 
                             + (1 << s_info.vec_exponent) + s_info.vec_fraction + 1;
    const int matrix_wr_times = (s_info.num_blocks + max_num_engines - 1) / max_num_engines;
    const double t_pipeline_tail = 1.0 * s_info.bsize / r_cfg.freq;
    const double t_xbar_wr = r_cfg.t_write_per_row * s_info.bsize;

    t_sim += (matrix_wr_times == 1)? t_xbar_wr : 0.0;

    process_time = get_time() - curr_time;
    cout << "###########################\n";

    curr_time = get_time();
    // r <- rhs
    blas_copy(n, rhs, 1, r, 1);

    // r <- (-1.0)*A*x0 + (1.0)*r
    sparse_spmv(n, nnz, -1.0,
                csrRowPtr, csrColIndex, csrVal,
                x0, 1.0, r, bsize);
    t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
    t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

    // r0_bar <- r
    blas_copy(n, r, 1, r0_bar, 1);

    k = 0;

    // res = r'*r
    res = blas_dot(n, r, 1, r, 1);

    printf("iteration = %3d, me = %e, res = %e\n", k, res / n, res);

    bool normal = (!isinf(res) && !isnan(res));

    for (k = 1; normal && (res > th) && k <= n_ite; ++k) {
        // rho_new = r0_bar' * r;
        rho_new = blas_dot(n, r0_bar, 1, r, 1);

        beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v);
        blas_axpby(n, -1.0 * omega, v, 1, 1.0, p, 1);
        blas_axpby(n, 1.0, r, 1, beta, p, 1);

        // convert the vector p
        if (vector2refloat != NULL) {
            vector2refloat(n, bsize, p, vec_exponent, vec_fraction);
        }

        // v = A * p;
        sparse_spmv(n, nnz, 1.0,
                    csrRowPtr, csrColIndex, csrVal,
                    p, 0.0, v, bsize);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        // alpha = rho_new / (r0_bar' * v);
        alpha = rho_new / blas_dot(n, r0_bar, 1, v, 1);

        // s = r - alpha * v;
        blas_copy(n, v, 1, s, 1);
        blas_axpby(n, 1.0, r, 1, -1.0 * alpha, s, 1);

        // convert the vector p
        if (vector2refloat != NULL) {
            vector2refloat(n, bsize, s, vec_exponent, vec_fraction);
        }

        // t = A * s;
        sparse_spmv(n, nnz, 1.0,
                    csrRowPtr, csrColIndex, csrVal,
                    s, 0.0, t, bsize);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        // omega = (t' * s) / (t' * t);
        omega = blas_dot(n, t, 1, s, 1) / blas_dot(n, t, 1, t, 1);

        // r = s - omega * t;
        blas_copy(n, t, 1, r, 1);
        blas_axpby(n, 1.0, s, 1, -1.0 * omega, r, 1);

        rho = rho_new;

        res = blas_dot(n, r, 1, r, 1);

        printf("iteration = %3d, me = %e, res = %e\n", k, res / n, res);
        normal = (!isinf(res) && !isnan(res));
    }

    delete[] r;
    delete[] r0_bar;
    delete[] v;
    delete[] p;
    delete[] s;
    delete[] t;

    process_time = get_time() - curr_time;
    if (s_info.platfrom != CPU) {
        process_time = t_sim;
    }
    return (normal? (k-1) : -1);
}

int CG_solver(int n,
              int nnz,
              int n_ite,
              double &process_time,
              int *csrRowPtr,
              int *csrColIndex,
              double *csrVal,
              double *rhs,
              double *x0,
              solverinfo & s_info) {
    double curr_time = get_time();

    void (*sparse_spmv)(int, int, double, int *, int *, double *, double *, double, double *, int) = s_info.sparse_spmv;
    const int bsize = s_info.bsize;
    void (*vector2refloat)(const int, const int, double *, const int exponent, const int) = s_info.vector2refloat;
    const int vec_exponent = s_info.vec_exponent;
    const int vec_fraction = s_info.vec_fraction;
    const double th = s_info.th;

    int k;
    double alpha_const, beta_const;
    double res, res_new;

    double *r = new double[n];
    double *p = new double[n];
    double *Ap = new double[n];

    const reraminfo r_cfg;
    double t_sim = 0.0;
    const int max_num_engines = (s_info.platfrom == CPU)? 1 : 
                                    (r_cfg.num_aval_xbar / 
                                    (4 * ((1 << s_info.matrix_exponent) + s_info.matrix_fraction + 1)));
    const int xbar_cycle = (1 << s_info.matrix_exponent) + s_info.matrix_fraction 
                             + (1 << s_info.vec_exponent) + s_info.vec_fraction + 1;
    const int matrix_wr_times = (s_info.num_blocks + max_num_engines - 1) / max_num_engines;
    const double t_pipeline_tail = 1.0 * s_info.bsize / r_cfg.freq;
    const double t_xbar_wr = r_cfg.t_write_per_row * s_info.bsize;

    t_sim += (matrix_wr_times == 1)? t_xbar_wr : 0.0;

    process_time = get_time() - curr_time;
    cout << "###########################\n";

    curr_time = get_time();
    // r <- rhs
    blas_copy(n, rhs, 1, r, 1);

    // r <- (-1.0)*A*x0 + (1.0)*r
    alpha_const = -1.0;
    beta_const = 1.0;
    sparse_spmv(n,
                nnz,
                alpha_const,
                csrRowPtr,
                csrColIndex,
                csrVal,
                x0,
                beta_const,
                r,
                bsize);
    t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
    t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

    // p <- r
    blas_copy(n, r, 1, p, 1);

    // convert the vector p
    if (vector2refloat != NULL) {
        vector2refloat(n, bsize, p, vec_exponent, vec_fraction);
    }

    k = 0;

    // res = r'*r
    res = blas_dot(n, r, 1, r, 1);

    printf("ite = %3d, me = %e, res = %e\n", k, res / n, res);

    bool normal = (!isinf(res) && !isnan(res));

    for (k = 1; normal && (res > th) && (k <= n_ite); ++k) {
        // Ap <- A*p
        sparse_spmv(n,
                    nnz,
                    1.0,
                    csrRowPtr,
                    csrColIndex,
                    csrVal,
                    p,
                    0.0,
                    Ap,
                    bsize);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        double pAp = blas_dot(n, p, 1, Ap, 1);

        alpha_const = res / pAp;

        // r <- (-alpha_const) * Ap + r
        blas_axpby(n, -1.0 * alpha_const, Ap, 1, 1.0, r, 1);

        res_new = blas_dot(n, r, 1, r, 1);
        beta_const = res_new / res;

        // p <- 1.0*r + beta_const*p
        blas_axpby(n, 1.0, r, 1, beta_const, p, 1);

        // convert the vector p
        if (vector2refloat != NULL) {
            vector2refloat(n, bsize, p, vec_exponent, vec_fraction);
        }

        res = res_new;
        printf("ite = %3d, me = %e, res = %e\n", k, res / n, res);
        normal = (!isinf(res) && !isnan(res));
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    process_time = get_time() - curr_time;
    if (s_info.platfrom != CPU) {
        process_time = t_sim;
    }
    return (normal? (k - 1) : -1);
}

inline double double_2_MP(double x,
                          const int exponent,
                          const int fraction) {
    union double_ull {
        double dx;
        unsigned long long ullx;
    };

    double_ull value;
    value.dx = x;
    unsigned long long bias[11] = {
        0ULL,   // 1-1
        1ULL,   // 2-1
        3ULL,   // 4-1
        7ULL,   // 8-1
        15ULL,  // 16-1
        31ULL,  // 32-1
        63ULL,  // 64-1
        127ULL, // 128-1
        255ULL, // 256-1
        511ULL, // 512-1
        1023ULL // 1024-1
    };

    unsigned long long sign_bit = (1ULL << 63) & value.ullx; // double sign
    unsigned long long frac_bits = ((value.ullx << 12) >> (12 + 52 - fraction)) << (52 - fraction);
    unsigned long long exp_bits = (value.ullx >> 52) & 2047ULL;

    if ((exp_bits > 0) && (exp_bits < 2047)) {
        if (exp_bits > 1023) {
            exp_bits -= 1023;
            if (exp_bits > bias[exponent - 1]) {
                exp_bits = bias[exponent - 1];
            }
            exp_bits += 1023;
        }
        else {
            exp_bits = 1023 - exp_bits;
            if (exp_bits > bias[exponent - 1] - 1) {
                exp_bits = bias[exponent - 1] - 1;
            }
            exp_bits = 1023 - exp_bits;
        }
        exp_bits = exp_bits << 52;
    }
    value.ullx = sign_bit | exp_bits | frac_bits;
    return value.dx;
}

double blas_dot_MP(int n,
                   double *x,
                   int inc_x,
                   double *y,
                   int inc_y,
                   const int exponent,
                   const int fraction) {
    double ans = 0.0;
    for (int i = 0; i < n; ++i) {
        ans += double_2_MP(x[i * inc_x] * y[i * inc_y], exponent, fraction);
        ans = double_2_MP(ans, exponent, fraction);
    }
    return ans;
}

void blas_axpby_MP(int n,
                   double alpha,
                   double *x,
                   int inc_x,
                   double beta,
                   double *y,
                   int inc_y,
                   const int exponent,
                   const int fraction) {
    for (int i = 0; i < n; ++i) {
        y[i * inc_y] = double_2_MP(
            double_2_MP(alpha * x[i * inc_x], exponent, fraction) +
                double_2_MP(beta * y[i * inc_y], exponent, fraction),
            exponent, fraction);
    }
}

void sparse_spmv_csr_MP(int m,
                        int nnz,
                        double alpha,
                        int *csrRowPtr,
                        int *csrColIndex,
                        double *csrVal,
                        double *x,
                        double beta,
                        double *y,
                        const int exponent,
                        const int fraction) {
    for (int i = 0; i < m; ++i) {
        double psum = 0.0;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
            psum += double_2_MP(csrVal[j] * x[csrColIndex[j]], exponent, fraction);
            psum = double_2_MP(psum, exponent, fraction);
        }
        y[i] = double_2_MP(
            double_2_MP(alpha * psum, exponent, fraction) + double_2_MP(beta * y[i], exponent, fraction), exponent, fraction);
    }
}

int CG_HMP_solver(int n,
                  int nnz,
                  int n_ite,
                  double &process_time,
                  int *csrRowPtr,
                  int *csrColIndex,
                  double *csrVal,
                  double *rhs,
                  double *x0,
                  solverinfo & s_info) {
    double curr_time = get_time();

    const int exponent = s_info.vec_exponent;
    const int fraction = s_info.vec_fraction;
    const double th = s_info.th;

    int k;
    double alpha_const, beta_const;
    double res, res_new;

    double *r = new double[n];
    double *p = new double[n];
    double *Ap = new double[n];

    const reraminfo r_cfg;
    double t_sim = 0.0;
    const int max_num_engines = (s_info.platfrom == CPU)? 1 : 
                                    (r_cfg.num_aval_xbar / 
                                    (4 * ((1 << s_info.matrix_exponent) + s_info.matrix_fraction + 1)));
    const int xbar_cycle = (1 << s_info.matrix_exponent) + s_info.matrix_fraction 
                             + (1 << s_info.vec_exponent) + s_info.vec_fraction + 1;
    const int matrix_wr_times = (s_info.num_blocks + max_num_engines - 1) / max_num_engines;
    const double t_pipeline_tail = 1.0 * s_info.bsize / r_cfg.freq;
    const double t_xbar_wr = r_cfg.t_write_per_row * s_info.bsize;

    t_sim += (matrix_wr_times == 1)? t_xbar_wr : 0.0;

    process_time = get_time() - curr_time;
    cout << "###########################\n";

    curr_time = get_time();
    // r <- rhs
    blas_copy(n, rhs, 1, r, 1);

    // r <- (-1.0)*A*x0 + (1.0)*r
    alpha_const = -1.0;
    beta_const = 1.0;
    sparse_spmv_csr_MP(n,
                       nnz,
                       alpha_const,
                       csrRowPtr,
                       csrColIndex,
                       csrVal,
                       x0,
                       beta_const,
                       r,
                       exponent,
                       fraction);
    t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
    t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

    // p <- r
    blas_copy(n, r, 1, p, 1);

    k = 0;

    // res = r'*r
    res = blas_dot(n, r, 1, r, 1);

    printf("ite = %3d, me = %e, res = %e\n", k, res / n, res);

    bool normal = (!isinf(res) && !isnan(res));

    for (k = 1; normal && (res > th) && (k <= n_ite); ++k) {
        // Ap <- A*p
        sparse_spmv_csr_MP(n,
                           nnz,
                           1.0,
                           csrRowPtr,
                           csrColIndex,
                           csrVal,
                           p,
                           0.0,
                           Ap,
                           exponent, fraction);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        double pAp = blas_dot_MP(n, p, 1, Ap, 1, exponent, fraction);

        alpha_const = double_2_MP(res / pAp, exponent, fraction);

        // r <- (-alpha_const) * Ap + r
        blas_axpby_MP(n, -1.0 * alpha_const, Ap, 1, 1.0, r, 1, exponent, fraction);

        res_new = blas_dot(n, r, 1, r, 1);
        beta_const = double_2_MP(res_new / res, exponent, fraction);

        // p <- 1.0*r + beta_const*p
        blas_axpby_MP(n, 1.0, r, 1, beta_const, p, 1, exponent, fraction);

        res = res_new;
        printf("ite = %3d, me = %e, res = %e\n", k, res / n, res);
        normal = (!isinf(res) && !isnan(res));
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    process_time = get_time() - curr_time;
    if (s_info.platfrom != CPU) {
        process_time = t_sim;
    }
    return (normal? (k - 1) : -1);
}

int BiCG_HMP_solver(int n,
                    int nnz,
                    int n_ite,
                    double &process_time,
                    int *csrRowPtr,
                    int *csrColIndex,
                    double *csrVal,
                    double *rhs,
                    double *x0,
                    solverinfo & s_info) {
    double curr_time = get_time();

    const int exponent = s_info.vec_exponent;
    const int fraction = s_info.vec_fraction;
    const double th = s_info.th;

    int k;
    double alpha_const, beta_const;
    double alpha = 1.0;
    double beta = 0.0;
    double omega = 1.0;
    double rho = 1.0;
    double rho_new = 1.0;
    double res = 0.0;

    double *r = new double[n];
    double *r0_bar = new double[n];
    double *v = new double[n];
    double *p = new double[n];
    double *s = new double[n];
    double *t = new double[n];

    for (int i = 0; i < n; ++i) {
        r[i] = 0.0;
        v[i] = 0.0;
        p[i] = 0.0;
        s[i] = 0.0;
        t[i] = 0.0;
    }

    const reraminfo r_cfg;
    double t_sim = 0.0;
    const int max_num_engines = (s_info.platfrom == CPU)? 1 : 
                                    (r_cfg.num_aval_xbar / 
                                    (4 * ((1 << s_info.matrix_exponent) + s_info.matrix_fraction + 1)));
    const int xbar_cycle = (1 << s_info.matrix_exponent) + s_info.matrix_fraction 
                             + (1 << s_info.vec_exponent) + s_info.vec_fraction + 1;
    const int matrix_wr_times = (s_info.num_blocks + max_num_engines - 1) / max_num_engines;
    const double t_pipeline_tail = 1.0 * s_info.bsize / r_cfg.freq;
    const double t_xbar_wr = r_cfg.t_write_per_row * s_info.bsize;

    t_sim += (matrix_wr_times == 1)? t_xbar_wr : 0.0;

    process_time = get_time() - curr_time;
    cout << "###########################\n";

    curr_time = get_time();
    // r <- rhs
    blas_copy(n, rhs, 1, r, 1);

    // r <- (-1.0)*A*x0 + (1.0)*r
    alpha_const = -1.0;
    beta_const = 1.0;
    sparse_spmv_csr_MP(n, nnz, alpha_const,
                       csrRowPtr, csrColIndex, csrVal,
                       x0, beta_const, r,
                       exponent, fraction);
    t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
    t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

    // r0_bar <- r
    blas_copy(n, r, 1, r0_bar, 1);

    k = 0;

    // res = r'*r
    res = blas_dot(n, r, 1, r, 1);

    printf("iteration = %3d, me = %e, res = %e\n", k, res / n, res);

    bool normal = (!isinf(res) && !isnan(res));

    for (k = 1; normal && (res > th) && (k <= n_ite); ++k) {
        // rho_new = r0_bar' * r;
        rho_new = blas_dot_MP(n, r0_bar, 1, r, 1, exponent, fraction);
        // rho_new = blas_dot(n, r0_bar, 1, r, 1);

        beta = double_2_MP((rho_new / rho) * (alpha / omega), exponent, fraction);
        // beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v);
        blas_axpby_MP(n, -1.0 * omega, v, 1, 1.0, p, 1, exponent, fraction);
        blas_axpby_MP(n, 1.0, r, 1, beta, p, 1, exponent, fraction);

        // v = A * p;
        sparse_spmv_csr_MP(n, nnz, 1.0,
                           csrRowPtr, csrColIndex, csrVal,
                           p, 0.0, v, exponent, fraction);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        // alpha = rho_new / (r0_bar' * v);
        alpha = double_2_MP(rho_new / blas_dot_MP(n, r0_bar, 1, v, 1, exponent, fraction), exponent, fraction);

        // s = r - alpha * v;
        blas_copy(n, v, 1, s, 1);
        blas_axpby_MP(n, 1.0, r, 1, -1.0 * alpha, s, 1, exponent, fraction);

        // t = A * s;
        sparse_spmv_csr_MP(n, nnz, 1.0,
                           csrRowPtr, csrColIndex, csrVal,
                           s, 0.0, t, exponent, fraction);
        t_sim += (matrix_wr_times == 1)? 0.0 : (t_xbar_wr * matrix_wr_times);
        t_sim += (r_cfg.t_comp * xbar_cycle + t_pipeline_tail) * matrix_wr_times;

        // omega = (t' * s) / (t' * t);
        omega = double_2_MP(blas_dot_MP(n, t, 1, s, 1, exponent, fraction) / blas_dot_MP(n, t, 1, t, 1, exponent, fraction),
                            exponent, fraction);

        // r = s - omega * t;
        blas_copy(n, t, 1, r, 1);
        blas_axpby_MP(n, 1.0, s, 1, -1.0 * omega, r, 1, exponent, fraction);

        rho = rho_new;

        res = blas_dot(n, r, 1, r, 1);
        printf("iteration = %3d, me = %e, res = %e\n", k, res / n, res);
        normal = (!isinf(res) && !isnan(res));
    }

    delete[] r;
    delete[] r0_bar;
    delete[] v;
    delete[] p;
    delete[] s;
    delete[] t;

    process_time = get_time() - curr_time;
    if (s_info.platfrom != CPU) {
        process_time = t_sim;
    }
    return (normal? (k - 1) : -1);
}