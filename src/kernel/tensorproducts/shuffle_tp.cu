#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 1024

using namespace std;

void ShuffleTensorProductImpl::exec_tensor_product(
        uint64_t num_products,
        float* L1_in,
        float* L2_in,
        float* L3_out) {

}
