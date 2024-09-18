#include "tensorproducts.hpp"

#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "gpu_util.hpp"
#include "jit.hpp"

using namespace std;

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 512

UnrollTPImpl::UnrollTPImpl(
    Representation &L1_i,
    Representation &L2_i,
    Representation &L3_i,
    std::string jit_kernel) :
        GenericTensorProductImpl(L1_i, L2_i, L3_i),
        jit(jit_kernel) { 
    
    jit.compile("loop_unroll_many_to_one", {});
}

void UnrollTPImpl::exec_tensor_product(
    uint64_t num_products,
    float* L1_in,
    float* L2_in,
    float* L3_out) {

    void *args[] = { &num_products, &L1_in, &L2_in, &L3_out }; 
    jit.execute(A100_SMS * 2, THREAD_BLOCK_SIZE, args);
}

