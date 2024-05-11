#include "CusparseSpdm.h"
#include <iostream>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cusparse.h>
#include <device_launch_parameters.h>

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

double CusparseSpdm::runKernelSpGEMM(float* A, int nRowsA, int nColsA, float* B, int nRowsB, int nColsB, float* C) {

    int* d_csr_offsets, * d_csr_columns;
    float* d_csr_values, * d_dense;
    CHECK_CUDA(cudaMalloc((void**)&d_csr_offsets,
        (nRowsA + 1) * sizeof(int)));

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
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nRowsA, nColsA, 0,
        d_csr_offsets, NULL, NULL,
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
    CHECK_CUDA(cudaMalloc((void**)&d_csr_columns, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_csr_values, nnz * sizeof(float)));
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matA, d_csr_offsets, d_csr_columns,
        d_csr_values));
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle0, matA0, matA,
        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
        dBuffer0));

    cusparseDnMatDescr_t matB, matC;
    cusparseHandle_t     handle = nullptr;
    float alpha = 1.0f;
    float beta = 0.0f;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, nRowsB, nColsB, nColsB, B,
        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, nRowsA, nColsB, nColsB, C,
        CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA0))
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle0));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0.0;
}