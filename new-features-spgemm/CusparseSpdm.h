#ifndef __CUSPARSESPDM_H__
#define __CUSPARSESPDM_H__

#include "SpGeMM.h"

class CusparseSpdm :
    public SpGeMM
{
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA,
        float* B, int nRowsB, int nColsB,
        float* C);
};

#endif

