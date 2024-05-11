#ifndef __UTILS_H__
#define __UTILS_H__

int randomInit(float* data, int size, float sparsity = 0.0);
void print_array(float* data, int w, int h);
void print_array(int* data, int w, int h);
void matrixMulCPU(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB);
void printDiff(float* data1, float* data2, int width, int height, int iListLength, float fListTol);

#endif