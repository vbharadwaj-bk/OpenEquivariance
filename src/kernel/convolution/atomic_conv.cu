#include "convolution.hpp"
#include "gpu_util.hpp"
#include "util.hpp"
#include <iostream>

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

struct Graph {
    uint32_t* rows,
    uint32_t* cols,
    uint64_t nnz,
    uint32_t node_count
}

struct LInfo {
    float* ptr;
    size_t row_len;
};

__global__ atomicConvolve(Linfo L1, Linfo L2, Linfo L3, Graph g) {
     
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_idx  = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;



}


void AtomicConvImpl::exec_conv(
        float* L1_in,
        float* L2_in,
        float* L3_out,
        uint32_t* rows,
        uint32_t* cols,
        uint64_t nnz,
        uint32_t node_count,
        bool disable_tensor_op) {

    if(! disable_tensor_op) {
        throw std::invalid_argument("AtomicConvolve does not support tensor contraction yet!");
    }

    atomicConvolve<<<round_up(edge_count, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
        {L1_in, L1.get_rep_length()},
        {L2_in, L2.get_rep_length()},
        {L3_out, L3.get_rep_length()},
        {rows, cols, nnz, node_count}
    );

    gpuErrchk( cudaGetLastError() );
}

/*
__global__ void espmm_v1(
    ESPMM_Context ctx,
    uint64_t edge_count,
    uint64_t* rows,
    uint64_t* cols,
    float* X_in,
    float* edge_features,
    float* X_out) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_idx  = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    if(global_warp_idx < edge_count) {
        uint64_t row = rows[global_warp_idx];
        uint64_t col = cols[global_warp_idx];

        float X_in_val = X_in[col * ctx.X_in_rowlen + lane_id];
        atomicAdd(X_out + row * ctx.X_out_rowlen + lane_id, X_in_val); 
    }
}
*/