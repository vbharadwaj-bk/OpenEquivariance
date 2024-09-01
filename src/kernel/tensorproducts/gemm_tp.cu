#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"
#include "buffer.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 1024

/*
* Each thread performs a Kronecker product independent of others.
* This kernel exhibits a bad memory access pattern. 
*/
__global__ void kronecker_kernel_v1(
        size_t num_products,
        float* L1_in,
        size_t L1_len,
        float* L2_in,
        size_t L2_len,
        float* kprods) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_products) { 
        float* L1_vec = L1_in + (idx * L1_stride);
        float* L2_vec = L2_in + (idx * L2_stride);
        float* kprod = kprods + (idx * L1_stride * L2_stride);

        for(int i = 0; i < L1_len; i++) {
            for(int j = 0; j < L2_len; j++) {
                kprod[i * L1_len + j] = L1_vec[i] * L2_vec[j];
            }
        } 
    }
}

void GemmTensorProductImpl::exec_tensor_product(
        uint64_t num_products,
        float* L1_in,
        float* L2_in,
        float* L3_out) {

    size_t L1_len = get_row_length(1);
    size_t L2_len = get_row_length(2);
    size_t L3_len = get_row_length(3);

    gpuErrchk( cudaMemset(L3_out, 0.0, L3_stride * num_products * sizeof(float)) ) 
    DeviceBuffer<float> kprods(num_products * get_row_length(1) * get_row_length(2));

    kronecker_kernel_v1<<<round_up(num_products, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
        num_products,
        L1_in,
        L1_len,
        L2_in,
        L2_len,
        kprods);

    gpuErrchk( cudaGetLastError() );
}

