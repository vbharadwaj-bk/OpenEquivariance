#include "device_prop.hpp"

#include <cuda_runtime.h>
#include "gpu_util.hpp"

DeviceProp::DeviceProp(int device_id) {
    cudaDeviceProp prop; 
    cudaGetDeviceProperties(&prop, device_id);
    name = std::string(prop.name);
    gpuErrchk(cudaDeviceGetAttribute(&maxSharedMemoryPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxSharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&warpsize, cudaDevAttrWarpSize, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id));
}