#pragma once

#include "gpu_util.hpp"

class DeviceProp {
public:
    cudaDeviceProp prop;
    int warpsize;
    int major, minor;
    int multiprocessrCount;
    int maxSharedMemPerBlock;

    DeviceProp(int device_id) {
        gpuErrchk(cudaGetDeviceProperties(&prop, device_id));
        warpsize = prop.warpSize;
        major = prop.major;
        minor = prop.minor;
        multiprocessrCount = prop.multiProcessorCount;

        cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    }
};
