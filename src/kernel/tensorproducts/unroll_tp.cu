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
    KernelLaunchConfig &config_i 
    ) :
        GenericTensorProductImpl(reps),
        jit(jit_kernel),
        config(config_i) {
    jit.compile("loop_unroll_many_to_one", {});

    if(config.smem > 0) {
        jit.set_max_smem(config.smem);
    }
}

void UnrollTPImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    void *args[] = { &num_products, &L1_in, &L2_in, &L3_out }; 
    jit.execute(config.num_blocks, config.num_threads, args, config.smem);
}

