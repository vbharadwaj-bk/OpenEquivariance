#include "convolution.hpp"
#include "gpu_util.hpp"
#include "util.hpp"
#include <iostream>

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 256

struct Graph {
    uint32_t* rows;
    uint32_t* cols;
    uint64_t nnz;
    uint32_t node_count;
};

struct Linfo {
    float* ptr;
    size_t row_len;
};

__global__ void atomicConvolve(Linfo L1, Linfo L2, Linfo L3, Graph g) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_idx  = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    if(global_warp_idx < g.nnz) {
        uint64_t row = g.rows[global_warp_idx];
        uint64_t col = g.cols[global_warp_idx];

        float* in_row = L1.ptr + col * L1.row_len;
        float* out_row = L3.ptr + row * L3.row_len;

        for(size_t i = 0; i < L1.row_len; i += THREADS_PER_WARP) {
            if(i + lane_id < L1.row_len) {
                float in_val = in_row[i + lane_id];
                atomicAdd(out_row + i + lane_id, in_val);
            }
        }
    }
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

    gpuErrchk( cudaMemset(L3_out, 0.0, L3.get_rep_length() * node_count * sizeof(float)) )

    atomicConvolve<<<round_up(nnz * THREADS_PER_WARP, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
        {L1_in, L1.get_rep_length()},
        {L2_in, L2.get_rep_length()},
        {L3_out, L3.get_rep_length()},
        {rows, cols, nnz, node_count}
    );

    gpuErrchk( cudaGetLastError() );
}