#include <cuda_runtime.h>
#include "gpu_util.hpp"
#include "buffer.hpp"


void* gpu_alloc (size_t size) {
    void* ptr;
    gpuErrchk( cudaMalloc((void**) &ptr, size ))
    return ptr;
}

void gpu_free (void* ptr) {
    gpuErrchk( cudaFree(ptr))
}

void copy_host_to_device (void* host, void* device, size_t size) {
    gpuErrchk( cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
}

void copy_device_to_host (void* host, void* device, size_t size) {
    gpuErrchk( cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}