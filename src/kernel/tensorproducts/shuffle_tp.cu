#include "tensorproducts.hpp"

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"
#include "jit.hpp"

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

struct Linfo {
    float* ptr;
    unsigned int stride; // Assume here that stride is equal to length of row 
};

const char* SHUFFLE_JIT_CODE = R"(

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

struct Linfo {
    float* ptr;
    unsigned int stride; // Assume here that stride is equal to length of row 
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

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (num_products + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
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

)";

#define EXECUTE_OPTION(max_lane_l, red_depth) { \
    if(this->max_lane_length == max_lane_l && this->reduction_depth == red_depth) { \
        executed_kernel = true; \
        shuffle_tp_kernel<max_lane_l, red_depth> \
            <<<A100_SMS * 2, THREAD_BLOCK_SIZE>>>( \
                num_products, \
                {L1_in, static_cast<uint32_t>(L1.get_rep_length())}, \
                {L2_in, static_cast<uint32_t>(L2.get_rep_length())}, \
                {L3_out, static_cast<uint32_t>(L3.get_rep_length())}, \
                warp_values.ptr, \
                l1_indices.ptr, \
                l2_indices.ptr, \
                red_lanes.ptr \
            ); \
    } \
}

ShuffleTensorProductImpl::ShuffleTensorProductImpl(
    RepTriple &reps,
    py::array_t<float> warp_values_py, 
    py::array_t<int> l1_indices_py, 
    py::array_t<int> l2_indices_py, 
    py::array_t<int> red_lanes_py) :
            GenericTensorProductImpl(reps),
            warp_values(warp_values_py),
            l1_indices(l1_indices_py),
            l2_indices(l2_indices_py),
            red_lanes(red_lanes_py),
            jit(SHUFFLE_JIT_CODE) { 

    // Just to get max lane length
    Buffer<float> warp_values_dummy(warp_values_py); 
    Buffer<int> red_lanes_dummy(red_lanes_py);

    max_lane_length = static_cast<int>(warp_values_dummy.shape[0]);
    reduction_depth = static_cast<int>(red_lanes_dummy.shape[0]);
    
    jit.compile("shuffle_tp_kernel", {max_lane_length, reduction_depth});
}

void ShuffleTensorProductImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out,
    float* weights) {

    Linfo L1_info = {L1_in, static_cast<uint32_t>(L1.get_rep_length())};
    Linfo L2_info = {L2_in, static_cast<uint32_t>(L2.get_rep_length())};
    Linfo L3_info = {L3_out, static_cast<uint32_t>(L3.get_rep_length())};
    void *args[] = { &num_products, &L1_info, &L2_info, &L3_info, &warp_values.ptr, &l1_indices.ptr, &l2_indices.ptr, &red_lanes.ptr };
    jit.execute(0, A100_SMS * 2, THREAD_BLOCK_SIZE, args);
}
