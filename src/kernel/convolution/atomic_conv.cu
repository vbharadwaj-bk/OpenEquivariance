#include "convolution.hpp"
#include <iostream>

using namespace std;

void AtomicConvImpl::exec_conv(
        float* L1_in,
        float* L2_in,
        float* L3_out,
        uint32_t* rows,
        uint32_t* cols,
        uint64_t nnz,
        uint32_t node_count,
        bool disable_tensor_op
        ) {

    cout << "Starting convolution!" << endl;
}
