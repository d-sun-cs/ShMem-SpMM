#include <stdio.h>
#include <helper_functions.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <chrono>
#include "utils.h"
#include "constants.h"
#include "SpGEMM.h"
#include "GCOOSpDM.h"
#include "G2ShmemAsyncGCOOSpDM.h"
#include "CublasGeMM.h"
#include "CusparseSpdm.h"
#include "DenseReuseSpmm.h"
#include "DenseReuseNoAsync.h"


void setArgumentInt(int argc, char** argv, const char* string_ref, size_t& target);
void setArgumentFloat(int argc, char** argv, const char* string_ref, float& target);
void timeGemm(
	size_t nRowsA, size_t nColsA,
	size_t nRowsB, size_t nColsB,
	float sparsity,
	SpGeMM* algo);
void benchSpGemm(
	float* A, size_t nRowsA, size_t nColsA,
	float* B, size_t nRowsB, size_t nColsB,
	float* C,
	SpGeMM* algo);

int main(int argc, char** argv) {
	printf("[Benchmark sparse gemm on GPU (Sparse A multiply dence B)] - Starting...\n\n\n");

	checkCudaErrors(cudaSetDevice(0));

	size_t nRowsA = N_ROWS_A;
	size_t nColsA = N_COLS_A;
	size_t nRowsB = N_ROWS_B;
	size_t nColsB = N_COLS_B;
	float sparsity = A_SPARSITY;

	if (checkCmdLineFlag(argc, (const char**)argv, "help") ||
		checkCmdLineFlag(argc, (const char**)argv, "?")) {
		printf("Usage -nRowsA=nRowsA\n");
		printf("      -nColsA=nColsA\n");
		printf("      -nRowsB=nRowsB\n");
		printf("      -nColsB=nColsB\n");
		printf("      -nColsB=nColsB\n");
		printf("      -sparsity=sparsity (Sparsity of matrix B\n");
		printf("nColsA should equal to nRowsB\n");
		exit(EXIT_SUCCESS);
	}

	setArgumentInt(argc, argv, "nRowsA", nRowsA);
	setArgumentInt(argc, argv, "nColsA", nColsA);
	setArgumentInt(argc, argv, "nRowsB", nRowsB);
	setArgumentInt(argc, argv, "nColsB", nColsB);


	printf("[Warmup By CublasGeMM and CusparseSpmm]\n");
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Number of SMs: %d\n", prop.multiProcessorCount);
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new CublasGeMM());
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new CusparseSpdm());
	printf("[Warpup By CublasGeMM  and CusparseSpmm over]\n\n\n\n\n\n");

	printf("[======Begin Formal Test======]\n\n");

	printf("[GCOOSpDM Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new GCOOSpDM());
	printf("[GCOOSpDM Test Over]\n\n\n");

	printf("[G2ShmemAsyncGCOOSpDM Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new G2ShmemAsyncGCOOSpDM());
	printf("[G2ShmemAsyncCGCOOSpDM Test Over]\n\n\n");

	printf("[AsyncDenseReuseSpmm Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new DenseReuseSpmm());
	printf("[AsyncReuseSpmm Test Over]\n\n\n");

	printf("[DenseReuseNoAsync Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new DenseReuseNoAsync());
	printf("[DenseReuseNoAsync Test Over]\n\n\n");

	printf("[CusparseSpdm Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new CusparseSpdm());
	printf("[CusparseSpdm Test Over]\n\n\n");

	printf("[CublasGeMM Test]\n");
	timeGemm(nRowsA, nColsA, nRowsB, nColsB, sparsity, new CublasGeMM());
	printf("[CublasGeMM Test Over]\n\n\n");

	printf("[======Formal Test Over======]\n\n\n\n\n\n");

	return 0;
}

void setArgumentInt(int argc, char** argv, const char* string_ref, size_t& target) {
	if (checkCmdLineFlag(argc, (const char**)argv, string_ref)) {
		target = getCmdLineArgumentInt(argc, (const char**)argv, string_ref);
	}
}

void setArgumentFloat(int argc, char** argv, const char* string_ref, float& target) {
	if (checkCmdLineFlag(argc, (const char**)argv, string_ref)) {
		target = getCmdLineArgumentFloat(argc, (const char**)argv, string_ref);
	}
}

void timeGemm(size_t nRowsA, size_t nColsA, size_t nRowsB, size_t nColsB, float sparsity, SpGeMM* algo) {

	float* A = (float*)malloc(nRowsA * nColsA * sizeof(float));
	float* B = (float*)malloc(nRowsB * nColsB * sizeof(float));
	float* C = (float*)malloc(nRowsA * nColsB * sizeof(float));
	float* reference = (float*)malloc(nRowsA * nColsB * sizeof(float));
	int nnz = 0;

	// construct sparse A 
	nnz = randomInit(A, nRowsA * nColsA, sparsity);
	if (DEBUG) {
		printf("A: \n");
		print_array(A, nColsA, nRowsA);
	}
	printf("nnz of A: %d, density: %.4f\n", nnz, nnz * 1.0 / (nRowsA * nColsA));
	// construct dense B
	randomInit(B, nRowsB * nColsB);
	if (DEBUG) {
		printf("B: \n");
		print_array(B, nColsB, nRowsB);
	}

	// compute C
	benchSpGemm(A, nRowsA, nColsA, B, nRowsB, nColsB, C, algo);
	if (DEBUG) {
		printf("C: \n");
		print_array(C, nColsB, nRowsA);
	}

	// matrixMulCPU
	if (CPU_TEST) {
		auto start = std::chrono::steady_clock::now();
		matrixMulCPU(reference, A, B, nRowsA, nColsA, nColsB);
		auto end = std::chrono::steady_clock::now();
		int timeUsed = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count());
		printf("Time(millisecond) of matrix mul CPU:%d\n", timeUsed / 1000);
		if (DEBUG) {
			printf("Reference: \n");
			print_array(reference, nColsB, nRowsA);
		}
		bool resGPU = sdkCompareL2fe(reference, C, nRowsA * nColsB, 1.0e-5f);
		if (resGPU != true) {
			printf("Test failed, diff: \n");
			printDiff(reference, C, nColsB, nRowsA, 100, 1.0e-5f);
		}
		else {
			printf("Test passed\n");
		}
	}
	else {
		printf("Not execute CPU test\n");
	}
	free(A);
	free(B);
	free(C);
}

void benchSpGemm(
	float* A, size_t nRowsA, size_t nColsA,
	float* B, size_t nRowsB, size_t nColsB,
	float* C,
	SpGeMM* algo) {
	if (DEBUG) {
		printf("A[%ld,%ld] * B[%ld,%ld]\n\n", nRowsA, nColsA, nRowsB, nColsB);
	}

	float* d_A, * d_B, * d_C;
	cudaEvent_t start, stop;

	checkCudaErrors(cudaMalloc((void**)&(d_A), (size_t)nRowsA * nColsA * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(d_B), (size_t)nRowsB * nColsB * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&(d_C), (size_t)nRowsA * nColsB * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_A, A, (size_t)nRowsA * nColsA * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, (size_t)nRowsB * nColsB * sizeof(float), cudaMemcpyHostToDevice));

	// warmup
	algo->runKernelSpGEMM(d_A, nRowsA, nColsA, d_B, nRowsB, nColsB, d_C);
	//algo->runKernelSpGEMM(A, nRowsA, nColsA, B, nRowsB, nColsB, C);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < BENCH_REPEAT; i++) {
		algo->runKernelSpGEMM(d_A, nRowsA, nColsA, d_B, nRowsB, nColsB, d_C);
		//algo->runKernelSpGEMM(A, nRowsA, nColsA, B, nRowsB, nColsB, C);
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);

	checkCudaErrors(cudaMemcpy(C, d_C, nRowsA * nColsB * sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	printf("Total time(millisecond) of GPU computation:%f\n", msecTotal / BENCH_REPEAT);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}