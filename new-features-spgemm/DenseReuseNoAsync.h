#pragma once
#include "SpGEMM.h"
class DenseReuseNoAsync :
    public SpGeMM
{
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA,
        float* B, int nRowsB, int nColsB,
        float* C);
};

#pragma once
