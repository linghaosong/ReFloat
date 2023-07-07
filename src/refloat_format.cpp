#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <float.h>
#include <omp.h>
#include <limits>
#include <math.h>

using std::cout;
using std::endl;
using std::min;

const unsigned long long BIAS_TABLE[11] = {
	0ULL << 52,	  // 1-1
	1ULL << 52,	  // 2-1
	3ULL << 52,	  // 4-1
	7ULL << 52,	  // 8-1
	15ULL << 52,  // 16-1
	31ULL << 52,  // 32-1
	63ULL << 52,  // 64-1
	127ULL << 52, // 128-1
	255ULL << 52, // 256-1
	511ULL << 52, // 512-1
	1023ULL << 52 // 1024-1
};

const unsigned long long BIAS_FLOAT = 1024ULL << 52;

void fp64torft(const int len,
			   const int bsize,
			   double *arr,
			   const int exponent = 8,
			   const int fraction = 23) {
	if (exponent < 1 || exponent > 11) {
		cout << "exponent should be in [1,11]\n";
		exit(1);
	}

	if (fraction < 0 || exponent > 52) {
		cout << "fraction should be in [0,52]\n";
		exit(1);
	}

	unsigned long long SIGN = 1ULL << 63;
	unsigned long long EXP_MASK = 2047ULL << 52;
	unsigned long long FRAC_MASK = 0xfffffffffffff;

	unsigned long long MASK_RMV_EXP = SIGN | ((FRAC_MASK << (52 - fraction)) & FRAC_MASK);

	for (int i = 0; i < len; ++i) {
		double value = arr[i];
		unsigned long long *p = reinterpret_cast<unsigned long long *>(&value);
		unsigned long long p_exp = (*p) & EXP_MASK;

		if (p_exp > BIAS_FLOAT + BIAS_TABLE[exponent - 1]) {
			p_exp = BIAS_FLOAT + BIAS_TABLE[exponent - 1];
		}

		if (p_exp < BIAS_FLOAT - (1ULL << 52) - BIAS_TABLE[exponent - 1]) {
			p_exp = BIAS_FLOAT - (1ULL << 52) - BIAS_TABLE[exponent - 1];
		}

		*p = ((*p) & MASK_RMV_EXP) | p_exp;
		arr[i] = value;
	}
}

void convet2refloat(int nnz_c,
					int num_ele_in_thisblock,
					double *bcooVal,
					const int exponent = 8,
					const int fraction = 23) {
	if (exponent < 1 || exponent > 11) {
		cout << "exponent should be in [1,11]\n";
		exit(1);
	}

	if (fraction < 0 || exponent > 52) {
		cout << "fraction should be in [0,52]\n";
		exit(1);
	}

	unsigned long long *p = new unsigned long long;
	unsigned long long SIGN = 1ULL << 63;
	unsigned long long EXP_MASK = 2047ULL << 52;
	unsigned long long FRAC_MASK = 0xfffffffffffff;

	unsigned long long MASK_RMV_FRAC = SIGN | EXP_MASK;
	unsigned long long MASK_RMV_EXP = SIGN | ((FRAC_MASK << (52 - fraction)) & FRAC_MASK);
	// unsigned long long MASK_RMV_EXP = SIGN | FRAC_MASK;

	double x_i = 0.0;
	int x_i_e_sum = 0;
	for (int i = nnz_c - num_ele_in_thisblock; i < nnz_c; ++i) {
		x_i = bcooVal[i];
		/**************** original
		unsigned long long * p = reinterpret_cast<unsigned long long*>(&x_i);
		*p = (*p) & EXP_MASK;
		****************/
		memcpy(p, &x_i, sizeof(x_i));
		*p = (*p) & EXP_MASK;
		memcpy(&x_i, p, sizeof(*p));

		x_i_e_sum += (int)round(log2(x_i));
	}

	if (num_ele_in_thisblock > 0) {
		unsigned long long Exp_Bias = (unsigned long long)(x_i_e_sum / num_ele_in_thisblock + 1023);
		Exp_Bias = Exp_Bias & 0x7ff;
		Exp_Bias = Exp_Bias << 52;

		for (int i = nnz_c - num_ele_in_thisblock; i < nnz_c; ++i) {
			double value = bcooVal[i];
			/**************** original
			unsigned long long * p = reinterpret_cast<unsigned long long*>(&value);
			unsigned long long p_exp = (*p) & EXP_MASK;
			****************/
			memcpy(p, &value, sizeof(value));
			unsigned long long p_exp = (*p) & EXP_MASK;

			if (p_exp > Exp_Bias + BIAS_TABLE[exponent - 1]) {
				p_exp = Exp_Bias + BIAS_TABLE[exponent - 1];
			}

			if (p_exp < Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1]) {
				p_exp = Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1];
			}

			*p = (*p) & MASK_RMV_EXP;
			*p = (*p) | p_exp;
			bcooVal[i] = value;
		}
	}
	delete p;
}

void vector2refloat(const int vec_length,
                    const int bsize,
                    double *vecVal,
                    const int exponent = -1,
                    const int fraction = -1) {
    if (exponent < 1 || exponent > 11) {
        cout << "vectro exponent should be in [1,11] or -1\n";
        exit(1);
    }

    if (fraction < 0 || exponent > 52) {
        cout << "vector fraction should be in [0,52]\n";
        exit(1);
    }

    // unsigned long long * p = new unsigned long long;
    unsigned long long SIGN = 1ULL << 63;
    unsigned long long EXP_MASK = 2047ULL << 52;
    unsigned long long FRAC_MASK = 0xfffffffffffff;
    unsigned long long MASK_RMV_FRAC = SIGN | EXP_MASK;
    unsigned long long MASK_MODIFY_FRAC = SIGN | EXP_MASK |
                                          ((FRAC_MASK << (52 - fraction)) & FRAC_MASK);
    unsigned long long MASK_RMV_EXP = SIGN | FRAC_MASK;

	#pragma omp parallel for num_threads(8)
    for (int i = 0; i < vec_length; ++i) {
        double value = vecVal[i];
        /**************** original
        unsigned long long * p = reinterpret_cast<unsigned long long*>(&value);
        *p = (*p) & MASK_MODIFY_FRAC;
        ****************/
        unsigned long long pp;
        unsigned long long *p = &pp;
        memcpy(p, &value, sizeof(value));
        *p = (*p) & MASK_MODIFY_FRAC;
        memcpy(&value, p, sizeof(*p));

        vecVal[i] = value;
    }

	#pragma omp parallel for num_threads(8)
    for (int b = 0; b < (vec_length + bsize - 1) / bsize; ++b) {
        int x_i_e_sum = 0;
        int num_ele_in_thisblock = 0;

        for (int i = b * bsize; i < min(vec_length, b * bsize + bsize); ++i) {
            if (vecVal[i] == 0.0)
                continue;

            double x_i = 0.0;
            ++num_ele_in_thisblock;
            x_i = vecVal[i];

            /**************** original
            unsigned long long * p = reinterpret_cast<unsigned long long*>(&x_i);
            *p = (*p) & EXP_MASK;
            ****************/
            unsigned long long pp;
            unsigned long long *p = &pp;
            memcpy(p, &x_i, sizeof(x_i));
            *p = (*p) & EXP_MASK;
            memcpy(&x_i, p, sizeof(*p));

            x_i_e_sum += (int)round(log2(x_i));
        }

        unsigned long long Exp_Bias = (unsigned long long)(x_i_e_sum / num_ele_in_thisblock + 1023);
        Exp_Bias = Exp_Bias & 0x7ff;
        Exp_Bias = Exp_Bias << 52;

        for (int i = b * bsize; i < min(vec_length, b * bsize + bsize); ++i) {
            if (vecVal[i] == 0.0)
                continue;

            double value = vecVal[i];

            /**************** original
            unsigned long long * p = reinterpret_cast<unsigned long long*>(&value);
            unsigned long long p_exp = (*p) & EXP_MASK;
            ****************/
            unsigned long long pp;
            unsigned long long *p = &pp;
            memcpy(p, &value, sizeof(value));
            unsigned long long p_exp = (*p) & EXP_MASK;

            if (p_exp > Exp_Bias + BIAS_TABLE[exponent - 1]) {
                p_exp = Exp_Bias + BIAS_TABLE[exponent - 1];
            }

            if (p_exp < Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1]) {
                p_exp = Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1];
            }

            *p = (*p) & MASK_RMV_EXP;
            *p = (*p) | p_exp;
            vecVal[i] = value;
        }
    }
    // delete p;
}

/******* default non omp 04 16 2019 ******/
void vector2refloatdefault(const int vec_length,
						   const int bsize,
						   double *vecVal,
						   const int exponent = -1,
						   const int fraction = -1) {
	if (exponent == -1 && fraction == -1) {
		return;
	}

	if (fraction < 0 || exponent > 52) {
		cout << "vector fraction should be in [0,52]\n";
		exit(1);
	}

	unsigned long long *p = new unsigned long long;
	unsigned long long SIGN = 1ULL << 63;
	unsigned long long EXP_MASK = 2047ULL << 52;
	unsigned long long FRAC_MASK = 0xfffffffffffff;
	unsigned long long MASK_RMV_FRAC = SIGN | EXP_MASK;
	unsigned long long MASK_MODIFY_FRAC = SIGN | EXP_MASK |
										  ((FRAC_MASK << (52 - fraction)) & FRAC_MASK);
	unsigned long long MASK_RMV_EXP = SIGN | FRAC_MASK;

	for (int i = 0; i < vec_length; ++i) {
		double value = vecVal[i];
		/**************** original
		unsigned long long * p = reinterpret_cast<unsigned long long*>(&value);
		*p = (*p) & MASK_MODIFY_FRAC;
		****************/
		memcpy(p, &value, sizeof(value));
		*p = (*p) & MASK_MODIFY_FRAC;
		memcpy(&value, p, sizeof(*p));

		vecVal[i] = value;
	}

	if (exponent == -1) {
		return;
	}

	if (exponent < 1 || exponent > 11) {
		cout << "vectro exponent should be in [1,11] or -1\n";
		exit(1);
	}

	double x_i = 0.0;
	int x_i_e_sum = 0;
	int num_ele_in_thisblock = 0;

	for (int b = 0; b < vec_length / bsize; ++b) {
		num_ele_in_thisblock = 0;
		x_i_e_sum = 0;
		for (int b = 0; b < (vec_length + bsize - 1) / bsize; ++b) {
			for (int i = b * bsize; i < min(vec_length, b * bsize + bsize); ++i) {
				if (vecVal[i] == 0.0)
					continue;

				++num_ele_in_thisblock;
				x_i = vecVal[i];

				/**************** original
				unsigned long long * p = reinterpret_cast<unsigned long long*>(&x_i);
				*p = (*p) & EXP_MASK;
				****************/
				memcpy(p, &x_i, sizeof(x_i));
				*p = (*p) & EXP_MASK;
				memcpy(&x_i, p, sizeof(*p));

				x_i_e_sum += (int)round(log2(x_i));
			}

			unsigned long long Exp_Bias = (unsigned long long)(x_i_e_sum / num_ele_in_thisblock + 1023);
			Exp_Bias = Exp_Bias & 0x7ff;
			Exp_Bias = Exp_Bias << 52;

			for (int i = b * bsize; i < min(vec_length, b * bsize + bsize); ++i) {
				if (vecVal[i] == 0.0)
					continue;

				double value = vecVal[i];

				/**************** original
				unsigned long long * p = reinterpret_cast<unsigned long long*>(&value);
				unsigned long long p_exp = (*p) & EXP_MASK;
				****************/
				memcpy(p, &value, sizeof(value));
				unsigned long long p_exp = (*p) & EXP_MASK;

				if (p_exp > Exp_Bias + BIAS_TABLE[exponent - 1]) {
					p_exp = Exp_Bias + BIAS_TABLE[exponent - 1];
				}

				if (p_exp < Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1]) {
					p_exp = Exp_Bias - (1ULL << 52) - BIAS_TABLE[exponent - 1];
				}

				*p = (*p) & MASK_RMV_EXP;
				*p = (*p) | p_exp;
				vecVal[i] = value;
			}
		}
	}
	delete p;
}

void csr2bcoo_refloat(int m,
					  int n,
					  int nnz,
					  int bsize,
					  int *csrRowPtr,
					  int *csrColIndex,
					  double *csrVal,
					  int *bcooRowIndex,
					  int *bcooColIndex,
					  double *bcooVal,
					  int &num_blocks,
					  const int exponent = 8,
					  const int fraction = 23) {
	num_blocks = 0;
	int nnz_c = 0;

	for (int row_start = 0; row_start < m; row_start += bsize) {
		int row_end = min(row_start + bsize, m);
		for (int col_start = 0; col_start < n; col_start += bsize) {
			int col_end = min(col_start + bsize, n);
			int num_ele_in_thisblock = 0;
			for (int i = row_start; i < row_end; ++i) {
				for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
					if (csrColIndex[j] < col_start)
						continue;

					if (csrColIndex[j] >= col_end)
						break;

					++num_ele_in_thisblock;
					bcooRowIndex[nnz_c] = i;
					bcooColIndex[nnz_c] = csrColIndex[j];
					bcooVal[nnz_c] = csrVal[j];
					++nnz_c;
				}
			}
			num_blocks += (num_ele_in_thisblock > 0);

			convet2refloat(nnz_c,
						   num_ele_in_thisblock,
						   bcooVal,
						   exponent,
						   fraction);
		}
	}

	/* order changed following is the default
		for(int col_start = 0; col_start < n; col_start += bsize) {
			int col_end = min(col_start + bsize, n);
			for(int row_start = 0; row_start < m; row_start += bsize) {
				int row_end = min(row_start + bsize, m);
				int num_ele_in_thisblock = 0;
				for (int i = row_start; i < row_end; ++i) {
					for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; ++j) {
						if (csrColIndex[j] < col_start) continue;
						if (csrColIndex[j] >= col_end) break;
						++num_ele_in_thisblock;
						bcooRowIndex[nnz_c] = i;
						bcooColIndex[nnz_c] = csrColIndex[j];
						bcooVal[nnz_c] = csrVal[j];
						++nnz_c;
					}
				}
				num_blocks += (num_ele_in_thisblock > 0);

				convet2refloat(nnz_c,
							   num_ele_in_thisblock,
							   bcooVal,
							   exponent,
							   fraction);
			}
		}
	*/
}

void csr2bcoo(int m,
			  int n,
			  int nnz,
			  int bsize,
			  int *csrRowPtr,
			  int *csrColIndex,
			  double *csrVal,
			  int *bcooRowIndex,
			  int *bcooColIndex,
			  double *bcooVal,
			  int &num_blocks) {
	num_blocks = 0;
	int nnz_c = 0;

	for (int row_start = 0; row_start < m; row_start += bsize) {
		int row_end = min(row_start + bsize, m);
		for (int col_start = 0; col_start < n; col_start += bsize) {
			int col_end = min(col_start + bsize, n);
			int num_ele_in_thisblock = 0;
			for (int i = row_start; i < row_end; ++i) {
				for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; ++j) {
					if (csrColIndex[j] < col_start)
						continue;

					if (csrColIndex[j] >= col_end)
						break;
						
					++num_ele_in_thisblock;
					bcooRowIndex[nnz_c] = i;
					bcooColIndex[nnz_c] = csrColIndex[j];
					bcooVal[nnz_c] = csrVal[j];
					++nnz_c;
				}
			}
			num_blocks += (num_ele_in_thisblock > 0);
		}
	}

	/* order changed following is the default
	for(int col_start = 0; col_start < n; col_start += bsize) {
		int col_end = min(col_start + bsize, n);
		for(int row_start = 0; row_start < m; row_start += bsize) {
			int row_end = min(row_start + bsize, m);
			int num_ele_in_thisblock = 0;
			for (int i = row_start; i < row_end; ++i) {
				for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; ++j) {
					if (csrColIndex[j] < col_start) continue;
					if (csrColIndex[j] >= col_end) break;
					++num_ele_in_thisblock;
					bcooRowIndex[nnz_c] = i;
					bcooColIndex[nnz_c] = csrColIndex[j];
					bcooVal[nnz_c] = csrVal[j];
					++nnz_c;
				}
			}
			num_blocks += (num_ele_in_thisblock > 0);
		}
	}
	*/
}

/************
The conversion of coo to bcoo can be integrated with the sorting process.
*************/

/*
const unsigned int BIAS_TABLE[8] = {
	0<<23, //1-1
	1<<23, //2-1
	3<<23, //4-1
	7<<23, //8-1
	15<<23,//16-1
	31<<23,//32-1
	63<<23,//64-1
	127<<23//128-1
};

const unsigned int BIAS_FLOAT = 128<<23;

void fp64torft(double * arr,
			   const int len,
			   const int exponent=8,
			   const int fraction=23) {
	if (exponent < 1 || exponent > 8) {
		cout << "exponent should be in [1,8]\n";
		exit(1);
	}

	if (fraction < 0 || exponent > 23) {
		cout << "fraction should be in [0,23]\n";
		exit(1);
	}

	unsigned int SIGN = 0x1 << 31;
	unsigned int EXP_MASK = 0xff << 23;
	unsigned int FRAC_MASK = 0x7fffff;

	int MASK_RMV_EXP = SIGN | ((FRAC_MASK << (23 - fraction)) & FRAC_MASK);

	for (int i = 0; i < len; ++i) {
		float value = (float) arr[i];
		unsigned int * p = reinterpret_cast<unsigned int*>(&value);
		unsigned int p_exp = (*p) & EXP_MASK;

		if (p_exp > BIAS_FLOAT + BIAS_TABLE[exponent-1]) {
			p_exp = BIAS_FLOAT + BIAS_TABLE[exponent-1];
		}
		if (p_exp < BIAS_FLOAT - (1<<23) - BIAS_TABLE[exponent-1]) {
			p_exp = BIAS_FLOAT - (1<<23) - BIAS_TABLE[exponent-1];
		}

		*p = ((*p) & MASK_RMV_EXP) | p_exp;
		arr[i] = (double) value;
	}
}
*/