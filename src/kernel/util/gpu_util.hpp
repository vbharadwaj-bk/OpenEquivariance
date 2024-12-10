#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

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

class GPUTimer {
    cudaEvent_t start_evt, stop_evt;

public:
    GPUTimer() {  
        cudaEventCreate(&start_evt);
        cudaEventCreate(&stop_evt);
    }

    void start() {
        cudaEventRecord(start_evt);
    }

    float stop_clock_get_elapsed() {
        float time_millis;
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        cudaEventElapsedTime(&time_millis, start_evt, stop_evt);
        return time_millis; 
    }
    
    ~GPUTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
};