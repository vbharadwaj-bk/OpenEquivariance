#include <iostream>
#include <cuda_runtime.h>
#include "espmm.hpp"
#include <cassert>

#DEFINE THREADS_PER_WARP 32

using namespace std;

/*
* This is a naive version of the code that uses atomics
* to perform the accumulation. Proof of concept to test
* the shuffle-add engine. 
*/

__global__ void espmm_v1(
    ESPMM_Context ctx,
    uint64_t edge_count,
    uint64_t* rows,
    uint64_t* cols,
    float* X_in,
    float* edge_features,
    float* X_out
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_idx  = idx / THREADS_PER_WARP;
    int lane_id = idx % THREADS_PER_WARP;

    if(global_warp_idx < edge_count) {
        uint64_t row = rows[global_warp_idx];
        uint64_t col = cols[global_warp_idx];

        float X_in_val = X_in[col * ctx.X_in_row_len + lane_id]; 
        X_out[row * ctx.X_out_row_len + lane_id] += X_in_val; 



    }

}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void check_cuda_device() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if(nDevices == 0) {
        cout << "Error, no CUDA-capable device detected!" << endl;
        exit(1);
    }
}

void equivariant_spmm_cpu(
        ESPMM_Context &context,
        uint64_t edge_count,
        uint64_t* rows,
        uint64_t* cols,
        float* X_in,
        float* X_out,
        float* edge_features) {

    check_cuda_device();

    uint64_t *d_rows, *d_cols;
    double *d_X_in, *d_edge_features, *d_X_out;

    gpuErrchk( cudaMalloc((void**)&d_rows, edge_count * sizeof(uint64_t)))
    gpuErrchk( cudaMalloc((void**)&d_cols, edge_count * sizeof(uint64_t)))
    gpuErrchk( cudaMalloc((void**)&d_X_in, context.node_count * context.X_in_row_len * sizeof(float)))
    gpuErrchk( cudaMalloc((void**)&d_edge_features, edge_count * context.edge_rowlen * sizeof(float)))
    gpuErrchk( cudaMalloc((void**)&d_X_out, context.node_count * context.X_out_row_len * sizeof(float))) 

    gpuErrchk( cudaMemcpy(d_rows, rows, edge_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_cols, cols, edge_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_X_in, X_in, context.node_count * context.X_in_row_len * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk( cudaMemcpy(d_edge_features, edge_features, edge_count * context.edge_rowlen * sizeof(float), cudaMemcpyHostToDevice))
    gpuErrchk( cudaMemset(d_X_out, 0, context.X_out_row_len * context.node_count * sizeof(float)))

    cout << "Computation goes here!" << endl;

    cudaMemcpy(X_out, d_X_out, context.node_count * context.X_out_row_len * sizeof(float), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaFree(d_rows))
    gpuErrchk( cudaFree(d_cols))
    gpuErrchk( cudaFree(d_X_in))
    gpuErrchk( cudaFree(d_edge_features))
    gpuErrchk( cudaFree(d_X_out))
}
