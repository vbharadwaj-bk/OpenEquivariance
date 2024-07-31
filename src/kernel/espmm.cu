#include <iostream>
#include <cuda_runtime.h>
#include "espmm.hpp"
#include <cassert>

using namespace std;

// This function accepts CPU pointers and copies
// its data to the GPU. 
/*void equivariant_spmm_cpu(
uint64_t node_count,
uint64_t edge_count,
uint64_t L1, 
uint64_t L2,
uint64_t L3,
uint64_t* row_ptr,
uint64_t* cols,
double* X_in,
double* X_out,
double* edge_features) {

}*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


uint64_t feature_length(uint64_t L) {
    return 2 * L + 1;
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
        double* X_in,
        double* X_out,
        double* edge_features) {

    check_cuda_device();

    uint64_t *d_rows, *d_cols;
    double *d_X_in, *d_edge_features, *d_X_out;

    gpuErrchk( cudaMalloc((void**)&d_rows, edge_count * sizeof(uint64_t)))
    gpuErrchk( cudaMalloc((void**)&d_cols, edge_count * sizeof(uint64_t)))
    gpuErrchk( cudaMalloc((void**)&d_X_in, context.node_count * feature_length(context.L1) * sizeof(double)))
    gpuErrchk( cudaMalloc((void**)&d_edge_features, edge_count * feature_length(context.L2) * sizeof(double)))
    gpuErrchk( cudaMalloc((void**)&d_X_out, context.node_count * feature_length(context.L3) * sizeof(double))) 

    gpuErrchk( cudaMemcpy(d_rows, rows, edge_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_cols, cols, edge_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(d_X_in, X_in, context.node_count * feature_length(context.L1) * sizeof(double), cudaMemcpyHostToDevice))
    gpuErrchk( cudaMemcpy(d_edge_features, edge_features, edge_count * feature_length(context.L2) * sizeof(double), cudaMemcpyHostToDevice))

    cout << "Computation goes here!" << endl;

    cudaMemcpy(X_out, d_X_out, context.node_count * feature_length(context.L3) * sizeof(double), cudaMemcpyDeviceToHost);

    gpuErrchk( cudaFree(d_rows))
    gpuErrchk( cudaFree(d_cols))
    gpuErrchk( cudaFree(d_X_in))
    gpuErrchk( cudaFree(d_edge_features))
    gpuErrchk( cudaFree(d_X_out))
}
