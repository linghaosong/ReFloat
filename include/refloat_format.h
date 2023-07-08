#ifndef REFLOAT_FORMAT
#define REFLOAT_FORMAT

void fp64torft(const int len,
               const int bsize,
               double *arr,
               const int exponent = 8,
               const int fraction = 23);

void convet2refloat(int nnz_c,
                    int num_ele_in_thisblock,
                    double *bcooVal,
                    const int exponent = 8,
                    const int fraction = 23);

void vector2refloat(const int vec_length,
                    const int bsize,
                    double *vecVal,
                    const int exponent = -1,
                    const int fraction = -1);

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
              int &num_blocks);

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
                      const int fraction = 23);

#endif