#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 1024

using namespace std;

struct Linfo{
    size_t stride;
    int mult;
    int l;
};


template<int mult1, int mult2>
__device__ __forceinline__ void multiplicity_outer_product_kernel(
    const float* __restrict__ L1_in,
    const uint8_t L1_idx, 
    const int L1, 
    const float* __restrict__ L2_in,
    const uint8_t L2_idx,
    const int L2,
    float* __restrict__ L3_out, 
    const uint8_t L3_idx,
    const int L3,
    const float value 
){  
    // Read in L1
    float L1_arr[mult1]; 
    #pragma unroll
    for (int i = 0; i < mult1; i++){
        L1_arr[i] = L1_in[i * (2 * L1 + 1) + L1_idx]; 
    }

    // Read in L2
    float L2_arr[mult2];
    #pragma unroll
    for (int j = 0; j < mult2; j++){
        L2_arr[j] = L2_in[j * (2 * L2 + 1) + L2_idx]; 
    }   

    // Intialize L3
    float L3_arr[mult1][mult2]; 

   
    // Calculations
    #pragma unroll
    for(int i = 0; i < mult1; i++){
        #pragma unroll
        for(int j = 0; j < mult2; j++){
            L3_arr[i][j] = L1_arr[i] * L2_arr[j] * value;
        }
    }

    // Writing Results Out
    int k = 0; 
    #pragma unroll
    for(int i = 0; i < mult1; i++){
        #pragma unroll
        for(int j = 0; j < mult2; j++){
            L3_out[k * (2 * L3 + 1) + L3_idx] += L3_arr[i][j]; 
            k++;  
        }
    }
}


template<int mult1, int mult2> 
__global__ void thread_tp_kernel(
        size_t num_products,
        const float* __restrict__ L1_in,
        Linfo  L1_info, 
        const float* __restrict__ L2_in,
        Linfo  L2_info, 
        float* __restrict__ L3_out,
        Linfo  L3_info,

        const size_t nnz,
        const uint8_t* __restrict__ coord1, 
        const uint8_t* __restrict__ coord2, 
        const uint8_t* __restrict__ coord3,
        const float* __restrict__ values) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    size_t L1_stride = L1_info.stride;  
    int L1_mult = L1_info.mult;
    int L1 = L1_info.l; 
    
    size_t L2_stride = L2_info.stride;
    int L2_mult = L2_info.mult; 
    int L2 = L2_info.l; 

    size_t L3_stride = L3_info.stride;
    int L3 = L3_info.l; 
    
    if(idx < num_products) {
        const float* L1_vec =  L1_in + (idx * L1_stride);
        const float* L2_vec =  L2_in + (idx * L2_stride);
        float* L3_vec = L3_out + (idx * L3_stride);

        for(int i = 0; i < nnz; i++) {
            multiplicity_outer_product_kernel<mult1, mult2>(
                L1_vec, coord1[i], L1, 
                L2_vec, coord2[i], L2,
                L3_vec, coord3[i], L3, 
                values[i]
            );
        }
    }
}

#define EXECUTE_OPTION(mult1, mult2) { \
    if(L1_info.mult == mult1 && L2_info.mult == mult2) { \
        executed_kernel = true; \
        thread_tp_kernel<mult1,mult2><<<round_up(num_products, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>( \
            num_products, \
            L1_in, \
            L1_info, \
            L2_in, \
            L2_info, \
            L3_out, \
            L3_info, \
            nnz, \
            coord1.ptr, \
            coord2.ptr, \
            coord3.ptr, \
            values.ptr); \
    } \
}

void MultiplicityOuterProductTensorProductImpl::exec_tensor_product(
        uint64_t num_products,
        float* __restrict__ L1_in,
        float* __restrict__ L2_in,
        float* __restrict__ L3_out) {

    size_t L1_stride = L1.get_rep_length(); 
    size_t L2_stride = L2.get_rep_length(); 
    size_t L3_stride = L3.get_rep_length(); 
    
    // This will eventually need to go when we add support for representations with sums
    gpuErrchk( cudaMemset(L3_out, 0.0, L3_stride * num_products * sizeof(float)) ) 
    size_t nnz = values.size;
    
    Linfo L1_info = {L1_stride, L1.mult(0), L1.type(0)};
    Linfo L2_info = {L2_stride, L2.mult(0), L2.type(0)};
    Linfo L3_info = {L3_stride, L3.mult(0), L3.type(0)};

    bool executed_kernel = false;

    EXECUTE_OPTION(1,1);
    EXECUTE_OPTION(1,2);
    EXECUTE_OPTION(2,1);
    EXECUTE_OPTION(2,2);
    EXECUTE_OPTION(4,4);
    EXECUTE_OPTION(8,8);
    EXECUTE_OPTION(16,16);

    cudaDeviceSynchronize();
    gpuErrchk( cudaGetLastError() );

    if(!executed_kernel) {
        throw std::runtime_error("Unsupported mult1, mult2: " + std::to_string(L1_info.mult) + ", " + std::to_string(L1_info.mult));
    }
}
