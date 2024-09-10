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
    int mult3_idx = 0; 
    #pragma unroll
    for(int mult1_idx = 0; mult1_idx < mult1; mult1_idx++){
        #pragma unroll
        for(int mult2_idx = 0; mult2_idx < mult2; mult2_idx++){

            int coord1 = mult1_idx * (2 * L1 + 1) + L1_idx; 
            int coord2 = mult2_idx * (2 * L2 + 1) + L2_idx; 
            int coord3 = mult3_idx * (2 * L3 + 1) + L3_idx;
    
            L3_out[coord3] += L1_in[coord1] * L2_in[coord2] * value;

            mult3_idx++;
        }
    }
}

__device__ __forceinline__ void get_correct_templated_kernel(
    const int mult1, 
    const int mult2, 
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
    switch(mult1){
        case 1:
            switch(mult2){
                case 1:
                    multiplicity_outer_product_kernel<1, 1>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                case 2:
                    multiplicity_outer_product_kernel<1, 2>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                default:
                    assert(false);
            }
            break;
        case 2:
            switch(mult2){
                case 1:
                    multiplicity_outer_product_kernel<2, 1>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                case 2:
                    multiplicity_outer_product_kernel<2, 2>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                default:
                    assert(false);
            } break; 
        case 4:
            switch(mult2){
                case 4:
                    multiplicity_outer_product_kernel<4, 4>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                default:
                    assert(false);
            } break;
        case 8: 
            switch(mult2){
                case 8:
                    multiplicity_outer_product_kernel<8, 8>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                default:
                    assert(false);
            } break;
        case 16:
            switch(mult2){
                case 16:
                    multiplicity_outer_product_kernel<16, 16>(L1_in, L1_idx, L1, L2_in, L2_idx, L2, L3_out, L3_idx, L3, value); break;
                default:
                    assert(false);
            } break;
            default:
                assert(false);
    }
}

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
            get_correct_templated_kernel(
                L1_mult, 
                L2_mult, 
                L1_vec, coord1[i], L1, 
                L2_vec, coord2[i], L2,
                L3_vec, coord3[i], L3, 
                values[i]
            );
        }
    }
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

    thread_tp_kernel<<<round_up(num_products, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
            num_products, 
            L1_in,
            L1_info, 
            L2_in,
            L2_info,
            L3_out,
            L3_info,

            nnz,
            coord1.ptr,
            coord2.ptr,
            coord3.ptr,
            values.ptr); 

    gpuErrchk( cudaGetLastError() );
}
