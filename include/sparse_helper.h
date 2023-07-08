#include <vector>
#include <iostream>
#include "mmio.h"

using std::cout;
using std::endl;
using std::vector;
using std::min;
using std::max;

#ifndef SPARSE_HELPER
#define SPARSE_HELPER

template <typename data_t>
struct rcv{
    int r;
    int c;
    data_t v;
};

enum MATRIX_FORMAT {CSR, CSC};

template <typename data_t>
struct edge{
    int col;
    int row;
    data_t attr;
    
    edge(int d = -1, int s = -1, data_t v = 0): col(d), row(s), attr(v) {}
    
    edge& operator=(const edge& rhs) {
        col = rhs.col;
        row = rhs.row;
        attr = rhs.attr;
        return *this;
    }
};

template <typename data_t>
int cmp_by_row_column(const void *aa,
                      const void *bb) {
    rcv<data_t> * a = (rcv<data_t> *) aa;
    rcv<data_t> * b = (rcv<data_t> *) bb;
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    return 0;
}

template <typename data_t>
int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv<data_t> * a = (rcv<data_t> *) aa;
    rcv<data_t> * b = (rcv<data_t> *) bb;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    return 0;
}

template <typename data_t>
void sort_by_fn(int nnz_s,
                vector<int> & cooRowIndex,
                vector<int> & cooColIndex,
                vector<data_t> & cooVal,
                int (* cmp_func)(const void *, const void *)) {
    auto rcv_arr = new rcv<data_t>[nnz_s];
    
    for(int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }
    
    qsort(rcv_arr, nnz_s, sizeof(rcv<data_t>), cmp_func);
    
    for(int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }
    
    delete [] rcv_arr;
}

void mm_init_read(FILE * f,
                  char * filename,
                  MM_typecode & matcode,
                  int & m,
                  int & n,
                  int & nnz) {
    //if ((f = fopen(filename, "r")) == NULL) {
    //        cout << "Could not open " << filename << endl;
    //        return 1;
    //}
    
    if (mm_read_banner(f, &matcode) != 0) {
        cout << "Could not process Matrix Market banner for " << filename << endl;
        exit(1);
    }
    
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
        cout << "Could not read Matrix Market format for " << filename << endl;
        exit(1);
    }
}

void load_S_matrix(FILE * f_A,
                   int nnz_mmio,
                   int & nnz,
                   vector<int> & cooRowIndex,
                   vector<int> & cooColIndex,
                   vector<double> & cooVal,
                   MM_typecode & matcode) {
    
    if (mm_is_complex(matcode)) {
        cout << "Redaing in a complex matrix, not supported yet!" << endl;
        exit(1);
    }
    
    if (!mm_is_symmetric(matcode)) {
        cout << "It's an NS matrix.\n";
    } else {
        cout << "It's an S matrix.\n";
    }
    
    int r_idx, c_idx;
    double value;
    int idx = 0;
    
    for (int i = 0; i < nnz_mmio; ++i) {
        if (mm_is_pattern(matcode)) {
            fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
            value = 1.0;
        }else {
            fscanf(f_A, "%d %d %lf\n", &r_idx, &c_idx, &value);
        }
        
        //unsigned int * tmpPointer_v = reinterpret_cast<unsigned int*>(&value);
        //unsigned int uint_v = *tmpPointer_v;
        
        uint64_t * tmpPointer_v = reinterpret_cast<uint64_t*>(&value);
        uint64_t uint_v = *tmpPointer_v;
        
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) { // report error
                cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << endl;
                exit(1);
            }
            
            cooRowIndex[idx] = r_idx - 1;
            cooColIndex[idx] = c_idx - 1;
            cooVal[idx] = value;
            idx++;
            
            if (mm_is_symmetric(matcode)) {
                if (r_idx != c_idx) {
                    cooRowIndex[idx] = c_idx - 1;
                    cooColIndex[idx] = r_idx - 1;
                    cooVal[idx] = value;
                    idx++;
                }
            }
        }
    }
    nnz = idx;
}

void read_suitsparse_matrix_FP64(char * filename_A,
                            vector<int> & elePtr,
                            vector<int> & eleIndex,
                            vector<double> & eleVal,
                            int & M,
                            int & K,
                            int & nnz,
                            const MATRIX_FORMAT mf=CSR) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE * f_A;
    
    if ((f_A = fopen(filename_A, "r")) == NULL) {
        cout << "Could not open " << filename_A << endl;
        exit(1);
    }
    
    mm_init_read(f_A, filename_A, matcode, M, K, nnz_mmio);
    
    if (!mm_is_coordinate(matcode)) {
        cout << "The input matrix file " << filename_A << "is not a coordinate file!" << endl;
        exit(1);
    }
    
    int nnz_alloc = (mm_is_symmetric(matcode))? (nnz_mmio * 2): nnz_mmio;
    cout << "Matrix A -- #row: " << M << " #col: " << K << endl;
    
    vector<int> cooRowIndex(nnz_alloc);
    vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    eleVal.resize(nnz_alloc);
    
    cout << "Loading input matrix A from " << filename_A << "\n";
    
    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, eleVal, matcode);
    
    fclose(f_A);
    
    if (mf == CSR) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_row_column<double>);
    }else if (mf == CSC) {
        sort_by_fn(nnz, cooRowIndex, cooColIndex, eleVal, cmp_by_column_row<double>);
    }else {
        cout << "Unknow format!\n";
        exit(1);
    }
    
    // convert to CSR/CSC format
    int M_K = (mf == CSR)? M : K;
    elePtr.resize(M_K+1);
    vector<int> counter(M_K, 0);
    
    if (mf == CSR) {
        for (int i = 0; i < nnz; i++) {
            counter[cooRowIndex[i]]++;
        }
    }else if (mf == CSC) {
        for (int i = 0; i < nnz; i++) {
            counter[cooColIndex[i]]++;
        }
    }else {
        cout << "Unknow format!\n";
        exit(1);
    }
    
    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
    }
    
    elePtr[0] = 0;
    for (int i = 1; i <= M_K; i++) {
        elePtr[i] = elePtr[i - 1] + counter[i - 1];
    }
    
    eleIndex.resize(nnz);
    if (mf == CSR) {
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooColIndex[i];
        }
    }else if (mf == CSC){
        for (int i = 0; i < nnz; ++i) {
            eleIndex[i] = cooRowIndex[i];
        }
    }
    
    if (mm_is_symmetric(matcode)) {
        //eleIndex.resize(nnz);
        eleVal.resize(nnz);
    }
}

template <typename data_t>
void cpu_spmv_CSR(const int M,
                  const int K,
                  const int NNZ,
                  const data_t ALPHA,
                  const vector<int> & CSRRowPtr,
                  const vector<int> & CSRColIndex,
                  const vector<data_t> & CSRVal,
                  const vector<data_t> & vec_X,
                  const data_t BETA,
                  vector<data_t> & vec_Y) {
    // A: sparse matrix, M x K
    // X: dense vector, K x 1
    // Y: dense vecyor, M x 1
    // output vec_Y = ALPHA * mat_A * vec_X + BETA * vec_Y
    // dense matrices: column major
    
    for (int i = 0; i < M; ++i) {
        data_t psum = 0;
        for (int j = CSRRowPtr[i]; j < CSRRowPtr[i+1]; ++j) {
            psum += CSRVal[j] * vec_X[CSRColIndex[j]];
        }
        vec_Y[i] = ALPHA * psum + BETA * vec_Y[i];
    }
}

void generate_edge_list_for_one_PE(const vector<edge<double>> & tmp_edge_list,
                                   vector<edge<double>> & edge_list,
                                   const int base_col_index,
                                   const int i_start,
                                   const int NUM_Row,
                                   const int NUM_PE,
                                   const int DEP_DIST_LOAD_STORE = 10){
    
    edge<double> e_empty = {-1, -1, 0.0};
    //vector<edge> scheduled_edges(NUM_Row);
    //std::fill(scheduled_edges.begin(), scheduled_edges.end(), e_empty);
    vector<edge<double>> scheduled_edges;
    
    //const int DEP_DIST_LOAD_STORE = 7;
    
    vector<int> cycles_rows(NUM_Row, -DEP_DIST_LOAD_STORE);
    int e_dst, e_src;
    float e_attr;
    for (unsigned int pp = 0; pp < tmp_edge_list.size(); ++pp) {
        e_src = tmp_edge_list[pp].col - base_col_index;
        //e_dst = tmp_edge_list[pp].row / 2 / NUM_PE;
        e_dst = tmp_edge_list[pp].row / NUM_PE;
        e_attr = tmp_edge_list[pp].attr;
        auto cycle = cycles_rows[e_dst] + DEP_DIST_LOAD_STORE;
        
        bool taken = true;
        while (taken){
            if (cycle >= ((int)scheduled_edges.size()) ) {
                scheduled_edges.resize(cycle + 1, e_empty);
            }
            auto e = scheduled_edges[cycle];
            if (e.row != -1)
                cycle++;
            else
                taken = false;
        }
        scheduled_edges[cycle].col = e_src;
        //scheduled_edges[cycle].row = e_dst * 2 + (tmp_edge_list[pp].row % 2);
        scheduled_edges[cycle].row = e_dst;
        scheduled_edges[cycle].attr = e_attr;
        cycles_rows[e_dst] = cycle;
    }
    
    int scheduled_edges_size = scheduled_edges.size();
    if (scheduled_edges_size > 0) {
        //edge_list.resize(i_start + scheduled_edges_size + DEP_DIST_LOAD_STORE - 1, e_empty);
        edge_list.resize(i_start + scheduled_edges_size, e_empty);
        for (int i = 0; i < scheduled_edges_size; ++i) {
            edge_list[i + i_start] = scheduled_edges[i];
        }
    }
}


void generate_edge_list_for_all_PEs(const vector<int> & CSCColPtr,
                                    const vector<int> & CSCRowIndex,
                                    const vector<double> & CSCVal,
                                    const int NUM_PE,
                                    const int NUM_ROW,
                                    const int NUM_COLUMN,
                                    const int WINDOE_SIZE,
                                    vector<vector<edge<double>> > & edge_list_pes,
                                    vector<int> & edge_list_ptr,
                                    const int DEP_DIST_LOAD_STORE = 10) {
    edge_list_pes.resize(NUM_PE);
    edge_list_ptr.resize((NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE + 1, 0);
    
    vector<vector<edge<double>> > tmp_edge_list_pes(NUM_PE);
    for (int i = 0; i < (NUM_COLUMN + WINDOE_SIZE - 1) / WINDOE_SIZE; ++i) {
        for (int p = 0; p < NUM_PE; ++p) {
            tmp_edge_list_pes[p].resize(0);
        }
        
        //fill tmp_edge_lsit_pes
        for (int col =  WINDOE_SIZE * i; col < min(WINDOE_SIZE * (i + 1), NUM_COLUMN); ++col) {
            for (int j = CSCColPtr[col]; j < CSCColPtr[col+1]; ++j) {
                //int p = (CSCRowIndex[j] / 2) % NUM_PE; 
                int p = CSCRowIndex[j] % NUM_PE;
                int pos = tmp_edge_list_pes[p].size();
                tmp_edge_list_pes[p].resize(pos + 1);
                tmp_edge_list_pes[p][pos] = edge<double>(col, CSCRowIndex[j], CSCVal[j]);
            }
        }
        
        //form the scheduled edge list for each PE
        for (int p = 0; p < NUM_PE; ++p) {
            int i_start = edge_list_pes[p].size();
            int base_col_index = i * WINDOE_SIZE;
            generate_edge_list_for_one_PE(tmp_edge_list_pes[p],
                                          edge_list_pes[p],
                                          base_col_index,
                                          i_start,
                                          NUM_ROW,
                                          NUM_PE,
                                          DEP_DIST_LOAD_STORE);
        }
        
        //insert bubules to align edge list
        int max_len = 0;
        for (int p = 0; p < NUM_PE; ++p) {
            max_len = max((int) edge_list_pes[p].size(), max_len);
        }
        for (int p = 0; p < NUM_PE; ++p) {
            edge_list_pes[p].resize(max_len, edge<double>(-1,-1,0.0));
        }
        
        //pointer
        edge_list_ptr[i+1] = max_len;
    }
    
}

template <typename data_t>
void CSC_2_CSR(int M,
               int K,
               int NNZ,
               const vector<int> & csc_col_Ptr,
               const vector<int> & csc_row_Index,
               const vector<data_t> & cscVal,
               vector<int> & csr_row_Ptr,
               vector<int> & csr_col_Index,
               vector<data_t> & csrVal) {
    csr_row_Ptr.resize(M + 1, 0);
    csrVal.resize(NNZ, 0.0);
    csr_col_Index.resize(NNZ, 0);
    
    for (int i = 0; i < NNZ; ++i) {
        csr_row_Ptr[csc_row_Index[i] + 1]++;
    }
    
    for (int i = 0; i < M; ++i) {
        csr_row_Ptr[i + 1] += csr_row_Ptr[i];
    }
    
    vector<int> row_nz(M, 0);
    for (int i = 0; i < K; ++i) {
        for (int j = csc_col_Ptr[i]; j < csc_col_Ptr[i + 1]; ++j) {
            int r = csc_row_Index[j];
            int c = i;
            auto v = cscVal[j];
            
            int pos = csr_row_Ptr[r] + row_nz[r];
            csrVal[pos] = v;
            csr_col_Index[pos] = c;
            row_nz[r]++;
        }
    }
}

#endif