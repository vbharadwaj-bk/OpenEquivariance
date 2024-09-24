#include "convolution.hpp"
#include "gpu_util.hpp"
#include "util.hpp"
#include <iostream>

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512
#define WARPS_PER_BLOCK THREAD_BLOCK_SIZE / THREADS_PER_WARP

#define A100_SMS 108

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
    __shared__ float buffers[WARPS_PER_BLOCK][ROW_LEN]; 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;
    int warp_loc = warp_id % (WARPS_PER_BLOCK);

    size_t warps_launched = blockDim.x * gridDim.x / THREADS_PER_WARP;
    size_t nnz_per_warp = (g.nnz + warps_launched - 1) / warps_launched;

    size_t start = nnz_per_warp * ((size_t) warp_id);
    size_t end = min(start + nnz_per_warp, g.nnz);

    // Zero out SMEM buffers 
    #pragma unroll
    for(size_t j = 0; j < ROW_LEN; j += THREADS_PER_WARP) {
        if(j + lane_id < ROW_LEN) {
            buffers[warp_loc][j + lane_id] = 0.0;
        }
    }

    bool firstSegment = true;
    for(int i = start; i < end; i++) {
        uint64_t row = g.rows[i];
        uint64_t col = g.cols[i];

        float* in_row = L1.ptr + col * L1.row_len;

        #pragma unroll
        for(size_t j = 0; j < ROW_LEN; j += THREADS_PER_WARP) {
            if(j + lane_id < ROW_LEN) {
                buffers[warp_loc][j + lane_id] += in_row[j + lane_id];
            }
        }

        // If changing rows and this is not the first segment or the last segment,
        // write directly to global memory 
        if(i < end - 1 && row != g.rows[i+1] && ! firstSegment) {
            float* out_row = L3.ptr + row * L3.row_len;
            #pragma unroll
            for(size_t j = 0; j < ROW_LEN; j += THREADS_PER_WARP) {
                if(j + lane_id < ROW_LEN) {
                    out_row[j + lane_id] = buffers[warp_loc][j + lane_id]; 
                    buffers[warp_loc][j + lane_id] = 0.0; // Zero out buffer for next accumulation
                }
            }
        }

        // If this is either the first or last segment, atomicAdd to the output row
        else if(i == end - 1 || firstSegment) {
            float* out_row = L3.ptr + row * L3.row_len;

            #pragma unroll
            for(size_t j = 0; j < ROW_LEN; j += THREADS_PER_WARP) {
                if(j + lane_id < ROW_LEN) {
                    atomicAdd(out_row + j + lane_id, buffers[warp_loc][j + lane_id]);
                    buffers[warp_loc][j + lane_id] = 0.0;
                }
            }

            firstSegment = false;
        }
    }
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

    gpuErrchk( cudaMemset(L3_out, 0.0, L3.get_rep_length() * node_count * sizeof(float)) )

    if(L1.get_rep_length() == 352) {
        SMConvolve<352> <<<A100_SMS * 2, THREAD_BLOCK_SIZE>>>(
            {L1_in, L1.get_rep_length()},
            {L2_in, L2.get_rep_length()},
            {L3_out, L3.get_rep_length()},
            {rows, cols, nnz, node_count}
        );
    } 

    gpuErrchk( cudaGetLastError() );
}