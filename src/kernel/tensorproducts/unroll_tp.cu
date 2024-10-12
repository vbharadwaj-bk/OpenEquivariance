#include "tensorproducts.hpp"

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"
#include "jit.hpp"

using namespace std;

UnrollTPImpl::UnrollTPImpl(
    RepTriple &reps,
    std::string jit_kernel,
    KernelLaunchConfig &forward_config_i,
    KernelLaunchConfig &backward_config_i) :
        GenericTensorProductImpl(reps),
        jit(jit_kernel),
        forward_config(forward_config_i),  
        backward_config(backward_config_i) {
    //jit.compile({"loop_unroll_many_to_one", "loop_unroll_backward"}, {{}, {}});
    jit.compile("loop_unroll_many_to_one", {}); 
}

void UnrollTPImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    if(forward_config.smem > 0) {
        jit.set_max_smem(forward_config.smem);
    }

    void *args[] = { &num_products, &L1_in, &L2_in, &L3_out }; 
    jit.execute(0, forward_config.num_blocks, forward_config.num_threads, args, forward_config.smem);
}


void UnrollTPImpl::backward(
        uint64_t num_products,
        float* L1_in, float* L1_grad,
        float* L2_in, float* L2_grad,
        float* weight, float* weight_grad,
        float* L3_grad) {

    if(backward_config.smem > 0) {
        jit.set_max_smem(backward_config.smem);
    }

    void *args[] = { &num_products, &L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad}; 
    jit.execute(1, backward_config.num_blocks, backward_config.num_threads, args, backward_config.smem);
} 

