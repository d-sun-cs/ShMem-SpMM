#include <cublas_v2.h>
#include <iostream>
#include "CublasGeMM.h"


double CublasGeMM::runKernelSpGEMM(float* A, int nRowsA, int nColsA, float* B, int nRowsB, int nColsB, float* C) {
	cublasHandle_t cublasHandle;
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasCreate(&cublasHandle);
	// It's exactly row-major C = A * B, although idk why it's correct
	cublasStatus_t status = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
		nColsB, nRowsA, nColsA,
		&alpha, B, nColsB, A, nColsA, &beta, C, nColsB);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Fail by cublasSgemm, status: %d\n", status);
		exit(1);
	}
	cublasDestroy(cublasHandle);
	return 0.0;
}