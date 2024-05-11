#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "constants.h"
#include "utils.h"
#include "GCOOSpDM.h"

#define ROW_PER_GROUP 4
#define WARP_SIZE 32 // fixed size
#define BLOCK_SIZE (2 * WARP_SIZE) // fixed size

void convertToGroupCOOFormat(float* A, int nRowsA, int nColsA,
	float*& pVals, int*& pRows, int*& pCols,
	int*& pGroupIndex, int*& pNnzPerGroup, int nGroup);
__global__ void cal_group_coo_format_nnz_kernel_cm(float* A, int nRowsA, int nColsA, int* pNnzPerGroup);
__global__ void prefix_sum_kernel2(int* src, int* dst, int n);
__global__ void convert_to_groupo_coo_format_kernel_cm(
	float* A, int nRowsA, int nColsA,
	float* pVals, int* pRows, int* pCols,
	int* pGroupIndex, int* pNnzPerGroup);
__global__ void sparse_dense_groupcoo_mat_mul_kernel(float* vals_A, int* cols_A, int* rows_A,
	int* groupIndex_A, int* nnzPerGroup_A,
	int nRowsA, int nColsA,
	float* B, int nRowsB, int nColsB,
	float* C);


double GCOOSpDM::runKernelSpGEMM(float* A, int nRowsA, int nColsA, float* B, int nRowsB, int nColsB, float* C) {
	float* pVals;
	int* pRows;
	int* pCols;
	int* pGroupIndex;
	int* pNnzPerGroup;
	int nGroup = (nRowsA + ROW_PER_GROUP - 1) / ROW_PER_GROUP;

	convertToGroupCOOFormat(A, nRowsA, nColsA,
		pVals, pRows, pCols,
		pGroupIndex, pNnzPerGroup, nGroup);
	cudaDeviceSynchronize();

	dim3 grid(nGroup, (nColsB + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 threadBlock(BLOCK_SIZE);
	sparse_dense_groupcoo_mat_mul_kernel << <grid, threadBlock >> > (
		pVals, pRows, pCols,
		pGroupIndex, pNnzPerGroup,
		nRowsA, nColsA,
		B, nRowsB, nColsB,
		C);
	cudaDeviceSynchronize();

	cudaFree(pVals);
	//cudaFree(pCols);
	cudaFree(pRows);
	cudaFree(pGroupIndex);
	return 0.0;
}

void convertToGroupCOOFormat(float* A, int nRowsA, int nColsA,
	float*& pVals, int*& pRows, int*& pCols,
	int*& pGroupIndex, int*& pNnzPerGroup, int nGroup) {
	/* Print nGroup */
	if (DEBUG) {
		printf("nGroup: %d\n\n", nGroup);
	}

	/* 1. Allocate pGroupIndex and pNnzPerGroup */
	if (DEBUG) {
		checkCudaErrors(cudaMallocManaged((void**)&pGroupIndex, sizeof(int) * (nGroup + 1)));
		checkCudaErrors(cudaMallocManaged((void**)&pNnzPerGroup, sizeof(int) * (nGroup + 1)));
	}
	else {
		checkCudaErrors(cudaMalloc((void**)&pGroupIndex, sizeof(int) * (nGroup + 1) * 2));
		// checkCudaErrors(cudaMalloc((void **) &pNnzPerGroup, sizeof(int) * (nGroup+1)));
		pNnzPerGroup = pGroupIndex + (nGroup + 1);
	}
	cudaMemset(pNnzPerGroup, 0, (nGroup + 1) * sizeof(int));

	/* 2. Calculate the number of non-zero elements in each group */
	dim3 gridCal(nGroup);
	dim3 tbCal(BLOCK_SIZE);
	cal_group_coo_format_nnz_kernel_cm << <gridCal, tbCal >> > (
		A, nRowsA, nColsA,
		pNnzPerGroup);
	cudaDeviceSynchronize();
	if (DEBUG) {
		printf("gpu pNnzPerGroup:\n");
		print_array(pNnzPerGroup, nGroup, 1);
	}

	/* 3. Calculate pGroupIndex with pNnzPerGroup */
	prefix_sum_kernel2 << <1, 1 >> > (pNnzPerGroup, pGroupIndex, nGroup + 1);
	cudaDeviceSynchronize();
	int* nnz_h = (int*)malloc(sizeof(int) * 1);
	checkCudaErrors(cudaMemcpy(nnz_h, pGroupIndex + nGroup, 1 * sizeof(int), cudaMemcpyDeviceToHost));
	int nnz = nnz_h[0];
	if (DEBUG) {
		printf("nnz: %d\n", nnz);
		printf("gpu pGroupIndex:\n");
		print_array(pGroupIndex, nGroup, 1);
	}

	/* 4. Allocate pVals, pRows and pCols */
	if (DEBUG) {
		checkCudaErrors(cudaMallocManaged((void**)&pVals, sizeof(float) * nnz));
		checkCudaErrors(cudaMallocManaged((void**)&pRows, sizeof(int) * nnz));
		checkCudaErrors(cudaMallocManaged((void**)&pCols, sizeof(int) * nnz));
	}
	else {
		checkCudaErrors(cudaMalloc((void**)&pVals, sizeof(float) * nnz));
		checkCudaErrors(cudaMalloc((void**)&pRows, sizeof(int) * nnz * 2));
		//checkCudaErrors(cudaMalloc((void **) &pCols, sizeof(int) * nnz));
		pCols = pRows + nnz;
	}

	/* 5. Calculate pVals, pRows and pCols */
	convert_to_groupo_coo_format_kernel_cm << <gridCal, tbCal >> > (A, nRowsA, nColsA,
		pVals, pRows, pCols,
		pGroupIndex, pNnzPerGroup);
	cudaDeviceSynchronize();
	if (DEBUG) {
		printf("vals:\n");
		print_array(pVals, nnz, 1);
		printf("rows:\n");
		print_array(pRows, nnz, 1);
		printf("cols:\n");
		print_array(pCols, nnz, 1);
	}

	free(nnz_h);
}

__global__ void cal_group_coo_format_nnz_kernel_cm(float* A, int nRowsA, int nColsA, int* pNnzPerGroup) {
	int startIdx = blockIdx.x * ROW_PER_GROUP;
	int nnz = 0;
	for (int i = threadIdx.x; i < nColsA; i += BLOCK_SIZE) {
		for (int j = 0; j < ROW_PER_GROUP; j++) {
			int row = j + startIdx;
			if (row >= nRowsA) {
				break;
			}
			float v = A[row * nColsA + i];
			if (v != 0.0) {
				nnz++;
			}
		}
	}
	typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	int aggregate = BlockReduceT(temp_storage).Sum(nnz);
	if (threadIdx.x == 0) {
		pNnzPerGroup[blockIdx.x] = aggregate;
	}
}

__global__ void prefix_sum_kernel2(int* src, int* dst, int n) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		dst[0] = 0;
		for (int i = 1; i < n; i++) {
			dst[i] = dst[i - 1] + src[i - 1];
		}
	}
}

__global__ void convert_to_groupo_coo_format_kernel_cm(
	float* A, int nRowsA, int nColsA,
	float* pVals, int* pRows, int* pCols,
	int* pGroupIndex, int* pNnzPerGroup) {

	int startIdx = blockIdx.x * ROW_PER_GROUP;
	int currGroupOffset = pGroupIndex[blockIdx.x];
	int cooIndex = currGroupOffset;
	float* currVals = pVals + cooIndex;
	int* currCols = pCols + cooIndex;
	int* currRows = pRows + cooIndex;

	__shared__ float sA[BLOCK_SIZE * ROW_PER_GROUP];
	typedef cub::BlockScan<int, BLOCK_SIZE> BlockScanT;
	__shared__ typename BlockScanT::TempStorage temp_storage;

	__shared__ int sNNz;
	sNNz = 0;
	__syncthreads();

	int end = (nColsA + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
	for (int i = threadIdx.x; i < end; i += BLOCK_SIZE) {
		int nnz = 0;
		int nnz_i = 0;
		/*
		*  Calculate the number of non-zero elements in the current block position.
		*  Each thread calculate the number of nnzs in its column(ROW_PER_GROUP).
		*/
		for (int j = 0; j < ROW_PER_GROUP; j++) {
			int row = j + startIdx;
			if (row < nRowsA && i < nColsA) {
				float v = A[row * nColsA + i];
				sA[j * BLOCK_SIZE + threadIdx.x] = v;
				if (v != 0.0) {
					nnz++;
				}
			}
		}
		BlockScanT(temp_storage).InclusiveSum(nnz, nnz_i);
		__syncthreads();
		BlockScanT(temp_storage).ExclusiveSum(nnz, nnz);

		float* vals = currVals + nnz;
		int* cols = currCols + nnz;
		int* rows = currRows + nnz;
		/*
		*  Fill in GCOO arrays in column-major way,
		*  so that continuous nnz in the same column can be accessed in the following algo.
		*/
		for (int j = 0; j < ROW_PER_GROUP; j++) {
			int row = j + startIdx;
			if (row >= nRowsA || i >= nColsA)
				break;
			float v = sA[j * BLOCK_SIZE + threadIdx.x];
			if (v != 0.0) {
				*(vals++) = v;
				*(rows++) = row;
				*(cols++) = i;
			}
		}
		if (threadIdx.x == BLOCK_SIZE - 1) {
			sNNz = nnz_i;
		}
		__syncthreads();
		currVals += sNNz;
		currCols += sNNz;
		currRows += sNNz;
	}
}

__global__ void sparse_dense_groupcoo_mat_mul_kernel(float* vals_A, int* rows_A, int* cols_A,
	int* groupIndex_A, int* nnzPerGroup_A,
	int nRowsA, int nColsA,
	float* B, int nRowsB, int nColsB,
	float* C) {
	int Cj = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	int Ci0 = blockIdx.x * ROW_PER_GROUP;
	float cx[ROW_PER_GROUP] = { 0.0f };
	int groupIdxOfCurrentBlock = groupIndex_A[blockIdx.x];
	int nnz = nnzPerGroup_A[blockIdx.x];
	float* currValsA = vals_A + groupIdxOfCurrentBlock;
	int* currColsA = cols_A + groupIdxOfCurrentBlock;
	int* currRowsA = rows_A + groupIdxOfCurrentBlock;

	__shared__ float sValsA[BLOCK_SIZE];
	__shared__ int sRowsA[BLOCK_SIZE];
	__shared__ int sColsA[BLOCK_SIZE];
	__shared__ int sNNz[1];

	int nIter = (BLOCK_SIZE + nnz - 1) / BLOCK_SIZE;
	int extra = nnz & (BLOCK_SIZE - 1);

	for (int i = 0; i < nIter; i++) {
		/*sColsA[threadIdx.x] = -1;
		sValsA[threadIdx.x] = 0.0;*/
		sNNz[0] = BLOCK_SIZE;
		__syncthreads();


		int valIdxStart = i * BLOCK_SIZE;
		int valIdx = valIdxStart + threadIdx.x;
		if (valIdx < nnz) {
			sValsA[threadIdx.x] = currValsA[valIdx];
			sRowsA[threadIdx.x] = currRowsA[valIdx];
			sColsA[threadIdx.x] = currColsA[valIdx];
		}
		else {
			sNNz[0] = extra;
		}
		__syncthreads();

		if (Cj < nColsB) {
			int rNNz = sNNz[0];
			int precol = -1;
			float b;
			for (int j = 0; j < rNNz; j++) {
				int col = sColsA[j];
				if (col != precol) {
					b = B[col * nColsB + Cj];
					precol = col;
				}
				float a = sValsA[j];
				int currRow = sRowsA[j];
				int index = currRow & (ROW_PER_GROUP - 1);
				// Each thread share a group of A's rows and process one column of B and C,
				// and each column contains ROW_PER_GROUP elements, represented by cx[idx]
				cx[index] += a * b;
			}
		}
		__syncthreads();
	}
	if (Cj < nColsB) {
		for (int i = 0; i < ROW_PER_GROUP && Ci0 + i < nRowsA; i++) {
			C[Cj + (Ci0 + i) * nColsB] = cx[i];
		}
	}

}