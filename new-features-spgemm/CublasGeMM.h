#ifndef __CUBLASGEMM_H__
#define __CUBLASGEMM_H__

#include "SpGeMM.h"
class CublasGeMM :
    public SpGeMM
{
public:
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA,
        float* B, int nRowsB, int nColsB,
        float* C);
};

#endif


