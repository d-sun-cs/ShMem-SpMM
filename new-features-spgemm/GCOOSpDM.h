#ifndef __GCOOSPDM_H__
#define __GCOOSPDM_H__

#include "SpGeMM.h"

class GCOOSpDM :
    public SpGeMM {
public:
    double runKernelSpGEMM(
        float* A, int nRowsA, int nColsA, 
        float* B, int nRowsB, int nColsB, 
        float* C);
};

#endif

