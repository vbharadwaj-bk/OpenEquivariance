#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

#define A100_SMS 108

using namespace std;

struct Linfo {
    float* ptr;
    uint32_t stride; // Assume here that stride is equal to length of row 
};

template <int MAX_LANE_LENGTH, int REDUCTION_DEPTH>
__global__ void shuffle_tp_kernel(
    size_t num_products,
    Linfo L1,
    Linfo L2,
    Linfo L3,
    float* warp_values_ptr,
    int* l1_indices_ptr,
    int* l2_indices_ptr,
    int* red_lanes_ptr) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    float values[MAX_LANE_LENGTH];
    int l1_indices[MAX_LANE_LENGTH];
    int l2_indices[MAX_LANE_LENGTH];
    int red_lanes[REDUCTION_DEPTH];

    // Load values into registers from global memory
    for(int i = 0; i < MAX_LANE_LENGTH; i++) {
        values[i] = warp_values_ptr[i * THREADS_PER_WARP + lane_id];
        l1_indices[i] = l1_indices_ptr[i * THREADS_PER_WARP + lane_id];
        l2_indices[i] = l2_indices_ptr[i * THREADS_PER_WARP + lane_id];
    }

    for(int i = 0; i < REDUCTION_DEPTH; i++) {
        red_lanes[i] = red_lanes_ptr[i * THREADS_PER_WARP + lane_id];
    }

    size_t warps_launched = blockDim.x * gridDim.x / 16;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = warp_id * nnz_per_warp;
    size_t end = min(start + nnz_per_warp, num_products);

    for(size_t i = start; i < end; i++) {
        float l1_vec = 0.0;
        float l2_vec = 0.0;
        float l3_vec = 0.0;

        // Step 1: Load vectors into warp lanes 
        if(lane_id < L1.stride) {
            float* l1_start = L1.ptr + i * L1.stride;
            l1_vec = l1_start[lane_id];
        }
        if(lane_id < L2.stride) {
            float* l2_start = L2.ptr + i * L2.stride;
            l2_vec = l2_start[lane_id];
        }

        // Step 2: Shuffle and multiply
        #pragma unroll
        for(int j = 0; j < MAX_LANE_LENGTH; j++) {
            float l1_val = __shfl_sync(0xFFFFFFFF, l1_vec, l1_indices[j]);
            float l2_val = __shfl_sync(0xFFFFFFFF, l2_vec, l2_indices[j]);
            l3_vec += l1_val * l2_val * values[j]; // TODO: Can have multiple accumulators 
        }

        // Step 3: Reduce if necessary
        #pragma unroll
        for(int j = 0; j < REDUCTION_DEPTH; j++) {
            float bcast_value = lane_id == 0 ? 0.0 : l3_vec;
            l3_vec += __shfl_sync(0xFFFFFFFF, bcast_value, red_lanes[j]); 
        }

        // Step 4: Store back 
        if(lane_id < L3.stride) {
            float* l3_start = L3.ptr + i * L3.stride;
            l3_start[lane_id] = l3_vec;
        }
    }
}

void ShuffleTensorProductImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {
        
    // Not really necessary
    gpuErrchk( cudaMemset(L3_out, 0.0, L3.get_rep_length() * num_products * sizeof(float)) )

    bool executed_kernel = false;

    if(this->max_lane_length == 4 && this->reduction_depth == 2) {
        executed_kernel = true;
        shuffle_tp_kernel<4, 3>
            <<<A100_SMS, THREAD_BLOCK_SIZE>>>(
                num_products,
                {L1_in, static_cast<uint32_t>(L1.get_rep_length())},
                {L2_in, static_cast<uint32_t>(L2.get_rep_length())},
                {L3_out, static_cast<uint32_t>(L3.get_rep_length())},
                warp_values.ptr,
                l1_indices.ptr,
                l2_indices.ptr,
                red_lanes.ptr
            ); 
    }

    cudaDeviceSynchronize();
    gpuErrchk( cudaGetLastError() );

    if(!executed_kernel) {
        throw std::runtime_error("Unsupported lane length and reduction depth: " + std::to_string(this->max_lane_length) + ", " + std::to_string(this->reduction_depth));
    }
}
