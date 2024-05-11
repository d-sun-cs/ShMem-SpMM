#ifndef __G2ShmemAsyncGCOOSPDM_H__
#define __G2ShmemAsyncGCOOSPDM_H__

#include "SpGeMM.h"

class G2ShmemAsyncGCOOSpDM :
    public SpGeMM {
public:
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA,
        float* B, int nRowsB, int nColsB,
        float* C);
};

#endif

