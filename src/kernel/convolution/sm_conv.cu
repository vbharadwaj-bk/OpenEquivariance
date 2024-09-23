#include "convolution.hpp"
#include "gpu_util.hpp"
#include "util.hpp"
#include <iostream>

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512
#define WARPS_PER_BLOCK THREAD_BLOCK_SIZE / THREADS_PER_WARP

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

template<int ROW_LEN>
__global__ void SMConvolve(Linfo L1, Linfo L2, Linfo L3, Graph g) {
    __shared__ float buffers[WARPS_PER_BLOCK][ROW_LEN][2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (g.nnz + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, g.nnz);
}

void SMConvImpl::exec_conv(
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

    // TODO: Shouldn't need this...
    //gpuErrchk( cudaMemset(L3_out, 0.0, L3.get_rep_length() * node_count * sizeof(float)) )

    if(L1.get_rep_length() == 352) {
        SMConvolve<352> <<<round_up(nnz * THREADS_PER_WARP, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
            {L1_in, L1.get_rep_length()},
            {L2_in, L2.get_rep_length()},
            {L3_out, L3.get_rep_length()},
            {rows, cols, nnz, node_count}
        );
    } 

    gpuErrchk( cudaGetLastError() );

    //cout << "Started convolution!" << endl;
}