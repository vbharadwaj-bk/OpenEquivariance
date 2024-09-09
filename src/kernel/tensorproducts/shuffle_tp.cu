#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

using namespace std;

struct Linfo {
    float* ptr;
    uint32_t stride;
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
    int global_warp_idx  = idx / THREADS_PER_WARP;
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


}

void ShuffleTensorProductImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

}
