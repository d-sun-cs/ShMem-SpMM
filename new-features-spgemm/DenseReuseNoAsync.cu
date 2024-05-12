#include "DenseReuseNoAsync.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cuda/pipeline>
//#include <cooperative_groups.h>

#define WARP_SIZE 32 
#define BLOCK_SIZE 128 // related with A_COLS_PER_BLOCK
#define A_COLS_PER_BLOCK 4 // related with BLOCK_SIZE
#define PIPELINE_STAGES 2
#define B_COLS_PER_BLOCK 128 // 4 * WARP_SIZE
//namespace cg = cooperative_groups;


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

__global__ void dense_reuse_matmul_kernel_no_async(int* d_csc_offsets, int* d_csc_rows,
    float* d_csc_values, int nnz, int nColsA, float* B, int nColsB, float* C) {

    int A_begin_col = A_COLS_PER_BLOCK * blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int A_col_idx = A_begin_col + warp_id;
    if (A_col_idx >= nColsA) {
        return;
    }
    int B_col_start_idx = blockIdx.y * B_COLS_PER_BLOCK;
    int B_col_end_idx = B_col_start_idx + B_COLS_PER_BLOCK > nColsB ?
        nColsB : B_col_start_idx + B_COLS_PER_BLOCK;
    int begin_offset = d_csc_offsets[A_col_idx];
    int end_offset = d_csc_offsets[A_col_idx + 1];
    if (begin_offset == end_offset) {
        return;
    }

    // __shared__ float Bs[PIPELINE_STAGES][BLOCK_SIZE];

    // auto pipe_B = cuda::make_pipeline();
    // const auto shape_float = cuda::aligned_size_t<alignof(float)>(sizeof(float));

    int B_begin_row = A_col_idx * nColsB;
    for (int i = 0, B_col_idx = lane_id + B_col_start_idx;
        // iStage = 0, bStage = lane_id + B_col_start_idx; 
        B_col_idx < B_col_end_idx;
        i++, B_col_idx += WARP_SIZE) {
        // for (; iStage < i + PIPELINE_STAGES && bStage < B_col_end_idx; iStage++, bStage += WARP_SIZE) {
        //     int j = iStage % PIPELINE_STAGES;
        //     pipe_B.producer_acquire();
        //     cuda::memcpy_async(&Bs[j][threadIdx.x], 
        //         &B[B_begin_row + bStage], shape_float, pipe_B);
        //     pipe_B.producer_commit();
        // }
        // int j = i % PIPELINE_STAGES;
        // pipe_B.consumer_wait();
        for (int k = begin_offset; k < end_offset; k++) {
            // atomicAdd(&C[d_csc_rows[k] * nColsB + B_col_idx],
            //     d_csc_values[k] * Bs[j][threadIdx.x]);
            atomicAdd(&C[d_csc_rows[k] * nColsB + B_col_idx],
                d_csc_values[k] * B[B_begin_row + B_col_idx]);
        }
        // pipe_B.consumer_release();
    }

}

double DenseReuseNoAsync::runKernelSpGEMM(float* A, int nRowsA, int nColsA,
    float* B, int nRowsB, int nColsB, float* C) {

    CHECK_CUDA(cudaMemset(C, 0, (size_t)nRowsA * nColsB * sizeof(float)));

    int* d_csc_offsets, * d_csc_rows;
    float* d_csc_values;
    CHECK_CUDA(cudaMalloc((void**)&d_csc_offsets,
        (nColsA + 1) * sizeof(int)));

    cusparseHandle_t     handle0 = nullptr;
    cusparseDnMatDescr_t matA0;
    cusparseSpMatDescr_t matA;
    void* dBuffer0 = NULL;
    size_t bufferSize0 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle0));

    // Create dense matrix A0
    CHECK_CUSPARSE(cusparseCreateDnMat(&matA0, nRowsA, nColsA, nColsA, A,
        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsc(&matA, nRowsA, nColsA, 0,
        d_csc_offsets, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
        handle0, matA0, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        &bufferSize0));
    CHECK_CUDA(cudaMalloc(&dBuffer0, bufferSize0));

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle0, matA0, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer0));
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp,
        &nnz));
    // allocate CSR column indices and values
    CHECK_CUDA(cudaMalloc((void**)&d_csc_rows, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csc_values, nnz * sizeof(float)));
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCscSetPointers(matA, d_csc_offsets, d_csc_rows,
        d_csc_values));
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle0, matA0, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer0));
    dim3 block(BLOCK_SIZE);
    dim3 grid((nColsA + A_COLS_PER_BLOCK - 1) / A_COLS_PER_BLOCK,
        (nColsB + B_COLS_PER_BLOCK - 1) / B_COLS_PER_BLOCK);
    dense_reuse_matmul_kernel_no_async << <grid, block >> > (d_csc_offsets, d_csc_rows, d_csc_values, nnz, nColsA, B, nColsB, C);
    return 0.0;
}