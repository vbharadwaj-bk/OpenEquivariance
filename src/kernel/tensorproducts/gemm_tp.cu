#include "tensorproducts.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "util.hpp"     // round_up
#include "gpu_util.hpp"
#include "buffer.hpp"
#include <cublasLt.h>

#define THREADS_PER_WARP 32
#define THREAD_BLOCK_SIZE 1024

/*
* Each thread performs a Kronecker product independent of others.
* This kernel exhibits a bad memory access pattern. 
*/
__global__ void kronecker_kernel_v1(
        size_t num_products,
        float* L1_in,
        size_t L1_len,
        float* L2_in,
        size_t L2_len,
        float* kprods) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < num_products) { 
        float* L1_vec = L1_in + (idx * L1_len);
        float* L2_vec = L2_in + (idx * L2_len);
        float* kprod = kprods + (idx * L1_len * L2_len);

        for(int i = 0; i < L1_len; i++) {
            for(int j = 0; j < L2_len; j++) {
                kprod[i * L2_len + j] = L1_vec[i] * L2_vec[j];
            }
        } 
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

void GemmTensorProductImpl::preprocess() {
    // cuBLASLt example taken from
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu 

    checkCublasStatus(cublasLtCreate(&ltHandle));

    size_t m = get_row_length(3);
    size_t n = num_products;
    size_t k = get_row_length(1) * get_row_length(2);

    size_t lda = k; 
    size_t ldb = k;
    size_t ldc = m;
    cublasOperation_t transa = CUBLAS_OP_T; 
    cublasOperation_t transb = CUBLAS_OP_N;

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb))); 

    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    int returnedResults                             = 0;
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
}

void GemmTensorProductImpl::exec_tensor_product(
        uint64_t num_products,
        float* L1_in,
        float* L2_in,
        float* L3_out) {

    size_t L1_len = get_row_length(1);
    size_t L2_len = get_row_length(2);
    size_t L3_len = get_row_length(3);

    gpuErrchk( cudaMemset(L3_out, 0.0, L3_len * num_products * sizeof(float)) )

    // Dynamic memory allocation here is expensive 

    kronecker_kernel_v1<<<round_up(num_products, THREAD_BLOCK_SIZE) / THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE>>>(
        num_products,
        L1_in,
        L1_len,
        L2_in,
        L2_len,
        kprods.ptr);

    float alpha = 1.0;
    float beta  = 0.0; 

    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     &alpha,
                                     cg_coeffs.ptr,
                                     Adesc,
                                     kprods.ptr,
                                     Bdesc,
                                     &beta,
                                     L3_out,
                                     Cdesc,
                                     L3_out,
                                     Cdesc,
                                     &heuristicResult.algo,
                                     workspace.ptr,
                                     workspaceSize,
                                     0));

    gpuErrchk( cudaGetLastError() );
    cudaDeviceSynchronize();

    //Buffer<float> kprods_host({kprods.size});
    //kprods.copy_to_host_buffer(kprods_host);
    //kprods_host.print();
}

GemmTensorProductImpl::~GemmTensorProductImpl() {
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));

    checkCublasStatus(cublasLtDestroy(ltHandle));
}