#include "DenseReuseSpmm.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cusparse.h>
#include <device_launch_parameters.h>
#include <cuda/pipeline>
//#include <cooperative_groups.h>

#define WARP_SIZE 32 
#define BLOCK_SIZE 128 // 4 * WARP_SIZE
#define A_COLS_PER_BLOCK 4 // BLOCK_SIZE / WARP_SIZE
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

__global__ void dense_reuse_matmul_kernel(int* d_csc_offsets, int* d_csc_rows,
    float* d_csc_values, int nnz, int nColsA, float *B, int nColsB, float *C) {

    int A_begin_col = A_COLS_PER_BLOCK * blockIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE; 
    int lane_id = threadIdx.x % WARP_SIZE;
    int A_col_idx = A_begin_col + warp_id;
    if (A_col_idx >= nColsA) {
        return;
    }
    int B_col_start_idx = blockIdx.y * B_COLS_PER_BLOCK;
    int B_col_end_idx = B_col_start_idx + B_COLS_PER_BLOCK > nColsB? 
        nColsB: B_col_start_idx + B_COLS_PER_BLOCK;
    // 同一个block中的不同warp读了相同数据，可以共享优化
    int begin_offset = d_csc_offsets[A_col_idx];
    int end_offset = d_csc_offsets[A_col_idx + 1];
    if (begin_offset == end_offset) {
        return;
    }

    __shared__ int As_rows[PIPELINE_STAGES][BLOCK_SIZE];
    __shared__ float As_vals[PIPELINE_STAGES][BLOCK_SIZE];
    __shared__ float Bs[PIPELINE_STAGES][BLOCK_SIZE];

    auto pipe_A = cuda::make_pipeline();
    auto pipe_B = cuda::make_pipeline();
    const auto shape_int = cuda::aligned_size_t<alignof(int)>(sizeof(int));
    const auto shape_float = cuda::aligned_size_t<alignof(float)>(sizeof(float));

    int Ai = 0;
    pipe_A.producer_acquire();
    if (begin_offset + lane_id < end_offset) {
        cuda::memcpy_async(&As_rows[0][threadIdx.x],
            &d_csc_rows[begin_offset + lane_id], shape_int, pipe_A);
        cuda::memcpy_async(&As_vals[0][threadIdx.x],
            &d_csc_values[begin_offset + lane_id], shape_float, pipe_A);
    }
    pipe_A.producer_commit();

    int B_begin_row = A_col_idx * nColsB;
    pipe_B.producer_acquire();
    if (B_col_start_idx + lane_id < B_col_end_idx) {
        cuda::memcpy_async(&Bs[0][threadIdx.x],
            &B[B_begin_row + B_col_start_idx + lane_id], shape_float, pipe_B);
    }
    pipe_B.producer_commit();

    for (int Bi = 0, B_col_idx = B_col_start_idx;
        B_col_idx < B_col_end_idx;
        Bi++, B_col_idx += WARP_SIZE) {

        pipe_B.consumer_wait();
        pipe_B.consumer_release();
        int Bj = Bi % PIPELINE_STAGES;
        for (int A_idx = begin_offset; 
            A_idx < end_offset; 
            A_idx += WARP_SIZE, Ai++) {

            if (A_idx + WARP_SIZE < end_offset) {
                pipe_A.producer_acquire();
                if (A_idx + WARP_SIZE + lane_id < end_offset) {
                    int Aj = (Ai + 1) % PIPELINE_STAGES;
                    cuda::memcpy_async(&As_rows[Aj][threadIdx.x],
                        &d_csc_rows[A_idx + WARP_SIZE + lane_id], shape_int, pipe_A);
                    cuda::memcpy_async(&As_vals[Aj][threadIdx.x],
                        &d_csc_values[A_idx + WARP_SIZE + lane_id], shape_float, pipe_A);
                }
                pipe_A.producer_commit();
            }
            else if (B_col_idx + WARP_SIZE < B_col_end_idx) {
                pipe_A.producer_acquire();
                if (begin_offset + lane_id < end_offset) {
                    int Aj = (Ai + 1) % PIPELINE_STAGES;
                    cuda::memcpy_async(&As_rows[Aj][threadIdx.x],
                        &d_csc_rows[begin_offset + lane_id], shape_int, pipe_A);
                    cuda::memcpy_async(&As_vals[Aj][threadIdx.x],
                        &d_csc_values[begin_offset + lane_id], shape_float, pipe_A);
                }
                pipe_A.producer_commit();
                pipe_B.producer_acquire();
                if (B_col_idx + WARP_SIZE + lane_id < B_col_end_idx) {
                    cuda::memcpy_async(&Bs[(Bi + 1) % PIPELINE_STAGES][threadIdx.x],
                        &B[B_begin_row + B_col_idx + WARP_SIZE + lane_id], shape_float, pipe_B);
                }
                pipe_B.producer_commit();
            }

            pipe_A.consumer_wait();
            pipe_A.consumer_release();

            //__syncwarp();
            if (B_col_idx + lane_id < B_col_end_idx) {
                int Aj = Ai % PIPELINE_STAGES;
                for (int k = 0; k < WARP_SIZE && k < end_offset - A_idx; k++) {
                    atomicAdd(&C[As_rows[Aj][warp_id * WARP_SIZE + k] * nColsB + B_col_idx + lane_id],
                        As_vals[Aj][warp_id * WARP_SIZE + k] * Bs[Bj][threadIdx.x]);
                    //int result = As_vals[Aj][warp_id * WARP_SIZE + k] * Bs[Bj][threadIdx.x];
                    //int result = As_vals[Aj][warp_id * WARP_SIZE + k];
                    //int result = Bs[Bj][threadIdx.x];
                    //C[As_rows[Aj][warp_id * WARP_SIZE + k] * nColsB + B_col_idx + lane_id] = 1234;
                }
            }
            __syncwarp();

            
        }
    }

}

double DenseReuseSpmm::runKernelSpGEMM(float* A, int nRowsA, int nColsA, 
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
    //dim3 grid((nColsA + A_COLS_PER_BLOCK - 1) / A_COLS_PER_BLOCK);
    dense_reuse_matmul_kernel << <grid, block >> > (d_csc_offsets, d_csc_rows, d_csc_values, nnz, nColsA, B, nColsB, C);
    return 0.0;
}