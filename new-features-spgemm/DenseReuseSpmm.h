#pragma once
#include "SpGeMM.h"
class DenseReuseSpmm :
    public SpGeMM
{
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA,
        float* B, int nRowsB, int nColsB,
        float* C);
};

