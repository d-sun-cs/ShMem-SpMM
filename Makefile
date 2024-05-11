main: main.cpp GCOOSpDM.cu G2ShmemAsyncGCOOSpDM.cu utils.cpp CublasGeMM.cpp CusparseSpdm.cu DenseReuseSpmm.cu DenseReuseNoAsync.cu constants.h
    nvcc -o main -lineinfo main.cpp GCOOSpDM.cu G2ShmemAsyncGCOOSpDM.cu utils.cpp CublasGeMM.cpp CusparseSpdm.cu DenseReuseSpmm.cu DenseReuseNoAsync.cu -I../cuda-samples/Common -arch=compute_80 -code=sm_80 -lcublas -lcusparse

run: 
    ./main

clean:
    rm -rf main

all:
    make clean; make; make run

ncu:
    ncu -o reporta6000 --import-source 1 --set full -f main