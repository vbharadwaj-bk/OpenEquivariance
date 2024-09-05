#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 1024

using namespace std;

__global__ void thread_tp_kernel(
        size_t num_products,
        float* L1_in,
        size_t L1_stride,
        size_t L1_mult,
        size_t L1, 
        float* L2_in,
        size_t L2_stride,
        size_t L2_mult,
        size_t L2,
        float* L3_out,
        size_t L3_stride,
        size_t L3,

        size_t nnz,
        uint8_t* coord1, 
        uint8_t* coord2, 
        uint8_t* coord3,
        float* values) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < num_products) {
        for(int mult1_idx = 0; mult1_idx < L1_mult; mult1_idx++){
            for(int mult2_idx = 0; mult2_idx < L2_mult; mult2_idx++){
                int mult3_idx = mult1_idx * L1_mult + mult2_idx; 
                float* L1_vec = L1_in + (idx * L1_stride) + (mult1_idx * (2 * L1 + 1));
                float* L2_vec = L2_in + (idx * L2_stride) + (mult2_idx * (2 * L2 + 1));
                float* L3_vec = L3_out + (idx * L3_stride) + (mult3_idx * (2 * L3 + 1));

                for(int i = 0; i < nnz; i++) {
                    L3_vec[coord3[i]] += L1_vec[coord1[i]] * L2_vec[coord2[i]] * values[i];
                }
            }
        }
    }
}

void ThreadTensorProductImpl::exec_tensor_product(
        uint64_t num_products,
        float* L1_in,
        float* L2_in,
        float* L3_out) {

    size_t L1_stride = L1.get_rep_length(); 
    size_t L2_stride = L2.get_rep_length(); 
    size_t L3_stride = L3.get_rep_length(); 

    gpuErrchk( cudaMemset(L3_out, 0.0, L3_stride * num_products * sizeof(float)) ) 
    size_t nnz = values.size;

    thread_tp_kernel<<<round_up(num_products, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
            num_products, 
            L1_in,
            L1_stride,
            L1.mult(0),
            L1.type(0), 
            L2_in,
            L2_stride,
            L2.mult(0),
            L2.type(0), 
            L3_out,
            L3_stride,
            L3.type(0),

            nnz,
            coord1.ptr,
            coord2.ptr,
            coord3.ptr,
            values.ptr); 

    gpuErrchk( cudaGetLastError() );
}
