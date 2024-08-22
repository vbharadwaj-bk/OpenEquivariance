#include <cuda_runtime.h>
#include "gpu_util.hpp"
#include "buffer.hpp"

template<typename T>
DeviceBuffer<T>::DeviceBuffer(uint64_t size) {
    this->size = size;
    gpuErrchk( cudaMalloc((void**) &ptr, size * sizeof(T)))
}

template<typename T>
DeviceBuffer<T>::~DeviceBuffer() {
    gpuErrchk( cudaFree(ptr))
}

template<typename T>
void DeviceBuffer<T>::copy_from_host_buffer(Buffer<T> &host) {
    // pass
}

template<typename T>
void DeviceBuffer<T>::copy_to_host_buffer(Buffer<T> &host) {
    // pass
}
