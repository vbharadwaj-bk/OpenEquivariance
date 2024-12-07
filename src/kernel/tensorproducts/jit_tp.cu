#include "tensorproducts.hpp"

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"
#include "jit.hpp"

using namespace std;

JITTPImpl::JITTPImpl(
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

void JITTPImpl::exec_tensor_product(
    uint64_t num_products,
    void* L1_in,
    void* L2_in,
    void* L3_out,
    void* weights) {

    void *args[] = { &num_products, &L1_in, &L2_in, &L3_out, &weights};
    jit.execute(0, forward_config.num_blocks, forward_config.num_threads, args, forward_config.smem);
}

void JITTPImpl::backward(
        size_t num_products,
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad) {

    void *args[] = { &num_products, &L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad}; 
    jit.execute(1, backward_config.num_blocks, backward_config.num_threads, args, backward_config.smem);
}