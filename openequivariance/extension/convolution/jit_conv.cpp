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

    vector<string> kernels = {"forward", "backward", "fixup_forward", "fixup_backward"};
    jit.compile(kernels, {{}, {}, {}, {}}); 

    if(forward_config.smem > 0) {
        jit.set_max_smem(0, forward_config.smem);
    }

    if(backward_config.smem > 0) {
        jit.set_max_smem(1, backward_config.smem);
    }
}

struct ConvData {
    void* rows;
    void* cols;
    unsigned long nnz;
    unsigned long node_count;
};

void JITConvImpl::exec_conv(
        void* L1_in,
        void* L2_in,
        void* weights, 
        void* L3_out,
        void* rows,
        void* cols,
        uint64_t nnz,
        uint64_t node_count,
        void* workspace) {

    ConvData conv_data = {rows, cols, nnz, node_count};

    void *args[] = {&L1_in, &L2_in, &weights, &L3_out, &conv_data, &workspace}; 
    jit.execute(0, args, forward_config);

    if(reinterpret_cast<uint64_t>(workspace) != 0) {
        void *fixup_args[] = {&workspace, &L3_out};
        jit.execute(2, fixup_args, forward_config);
    }
} 

void JITConvImpl::backward(
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad,
        void* rows, void* cols,
        uint64_t nnz, uint64_t node_count,
        void* workspace,
        void* transpose_perm) {

    ConvData conv_data = {rows, cols, nnz, node_count};
    void *args[] = {&L1_in, &L1_grad, &L2_in, &L2_grad, &weight, &weight_grad, &L3_grad, &conv_data, &workspace, &transpose_perm};
    jit.execute(1, args, backward_config);

    if(reinterpret_cast<uint64_t>(workspace) != 0) {
        void *fixup_args[] = {&workspace, &L1_grad};
        jit.execute(3, fixup_args, backward_config);
    }
}
