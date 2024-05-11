#ifndef __SPGEMM_H__
#define __SPGEMM_H__ 

class SpGeMM {
public:
	virtual double runKernelSpGEMM(
		float* A, int nRowsA, int nColsA, 
		float* B, int nRowsB, int nColsB, 
		float* C) = 0;
};

#endif

