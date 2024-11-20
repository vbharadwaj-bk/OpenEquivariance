#include "convolution.hpp"

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"
#include "jit.hpp"

using namespace std;

JITConvImpl::JITConvImpl
(
    std::string jit_kernel,
    KernelLaunchConfig &forward_config_i,
    KernelLaunchConfig &backward_config_i) :
        jit(jit_kernel),
        forward_config(forward_config_i),  
        backward_config(backward_config_i) {

    vector<string> kernels = {"forward", "backward"};
    jit.compile(kernels, {{}, {}}); 

    if(forward_config.smem > 0) {
        jit.set_max_smem(0, forward_config.smem);
    }

    if(backward_config.smem > 0) {
        jit.set_max_smem(1, backward_config.smem);
    }
}

struct ConvData {
    unsigned int* rows;
    unsigned int* cols;
    unsigned long nnz;
    unsigned int node_count;
};

void JITConvImpl::exec_conv(
        float* L1_in,
        float* L2_in,
        float* weights,
        float* L3_out,
        uint32_t* rows,
        uint32_t* cols,
        uint64_t nnz,
        uint32_t node_count,
        bool disable_tensor_op
        ) {

    ConvData conv_data = {rows, cols, nnz, node_count};

    void *args[] = {&L1_in, &L2_in, &weights, &L3_out, &conv_data, &disable_tensor_op}; 
    jit.execute(0, forward_config.num_blocks, forward_config.num_threads, args, forward_config.smem);
} 

void JITConvImpl::backward(
        float* L1_in, float* L1_grad,
        float* L2_in, float* L2_grad,
        float* weight, float* weight_grad,
        float* L3_grad,
        uint32_t* rows, uint32_t* cols,
        uint64_t nnz, uint32_t node_count,
        bool disable_tensor_op) {

    ConvData conv_data = {rows, cols, nnz, node_count};
    void *args[] = {&L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad, &conv_data, &disable_tensor_op};
    jit.execute(1, backward_config.num_blocks, backward_config.num_threads, args, backward_config.smem);
}
