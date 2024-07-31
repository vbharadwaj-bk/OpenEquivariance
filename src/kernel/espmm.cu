#include <iostream>
#include <cuda_runtime.h>
#include "espmm.hpp"

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

uint64_t feature_length(uint64_t L) {
    return 2 * L + 1;
}


void equivariant_spmm_cpu(
        ESPMM_Context &context,
        uint64_t edge_count,
        uint64_t* rows,
        uint64_t* cols,
        double* X_in,
        double* X_out,
        double* edge_features) {

    uint64_t *d_rows, *d_cols;
    double *d_X_in, *d_X_out;

    cudaMalloc((void**)&d_rows, edge_count * sizeof(uint64_t));
    cudaMalloc((void**)&d_cols, edge_count * sizeof(uint64_t));
    cudaMalloc((void**)&d_X_in, context.node_count * feature_length(context.L1) * sizeof(double));
    cudaMalloc((void**)&d_edge_features, edge_count * feature_length(context.L2) * sizeof(double));
    cudaMalloc((void**)&d_X_out, context.node_count * feature_length(context.L3) * sizeof(double));

    cudaMemcpy(rows, d_rows, edge_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cols, d_cols, edge_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(X_in, d_X_in, context.node_count * feature_length(context.L1) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(edge_features, d_edge_features, edge_count * feature_length(context.L2) * sizeof(double), cudaMemcpyDeviceToHost);
    
    



    cudaMemcpy(d_X_out, X_out, context.node_count * feature_length(context.L3) * sizeof(double), cudaMemcpyHostToDevice);
    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_X_in);
    cudaFree(d_edge_features);
    cudaFree(d_X_out);
}
