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


void equivariant_spmm_cpu(
        ESPMM_Context &context,
        uint64_t edge_count,
        uint64_t* row_ptr,
        uint64_t* cols,
        double* X_in,
        double* X_out,
        double* edge_features) {
    std::cout << "Hello world 7!" << std::endl;
}
