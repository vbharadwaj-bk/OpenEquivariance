#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cublasLt.h>
#include <cstdio>
#include <

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void check_cuda_device() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(nDevices == 0) {
        cout << "Error, no CUDA-capable device detected!" << endl;
        exit(1);
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}