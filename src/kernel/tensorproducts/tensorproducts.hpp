#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cublasLt.h>
#include <string>

#include "buffer.hpp"
#include "representation.hpp"
#include "jit.hpp"

class __attribute__ ((visibility ("default"))) GenericTensorProductImpl {
public:
    RepTriple reps;
    Representation &L1;
    Representation &L2;
    Representation &L3;

    bool record_internal_stats = false;

    GenericTensorProductImpl(
        RepTriple &reps_i) :
        reps(reps_i),
        L1(reps.L1), L2(reps.L2), L3(reps.L3)
        { }

    virtual void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights) = 0;
            
    // Executes function with CPU inputs from Python. Issues
    // memcpy to / from device. This function fills the weight matrix
    // with ones for now. 
    void exec_tensor_product_cpu(
            py::array_t<float> L1_in_py,
            py::array_t<float> L2_in_py,
            py::array_t<float> L3_out_py,
            py::array_t<float> weights_py) {
        
        // To get batch dimension 
        Buffer<float> L3_out_host(L3_out_py);
        auto batch_dim = L3_out_host.shape[0];

        // Copies data to device 
        DeviceBuffer<float> L1_in(L1_in_py);
        DeviceBuffer<float> L2_in(L2_in_py);
        DeviceBuffer<float> L3_out(L3_out_host.size());
        DeviceBuffer<float> weights(weights_py);

        exec_tensor_product(batch_dim, L1_in.ptr, L2_in.ptr, L3_out.ptr, weights.ptr);
        L3_out.copy_to_host_buffer(L3_out_host);
    }

    virtual void backward(
            size_t num_products,
            float* L1_in, float* L1_grad,
            float* L2_in, float* L2_grad,
            float* weight, float* weight_grad,
            float* L3_grad) {

        throw std::logic_error("Backward pass not implemented yet!");
    }

    void backward_cpu(
            py::array_t<float> L1_in_py, py::array_t<float> L1_grad_py,
            py::array_t<float> L2_in_py, py::array_t<float> L2_grad_py,
            py::array_t<float> weight_py, py::array_t<float> weight_grad_py,
            py::array_t<float> L3_grad_py) {

        Buffer<float> L1_grad_host(L1_grad_py);
        Buffer<float> L2_grad_host(L2_grad_py);
        Buffer<float> L3_grad_host(L3_grad_py);
        Buffer<float> weight_grad_host(weight_grad_py);

        // Copies data to device 
        DeviceBuffer<float> L1_in(L1_in_py);
        DeviceBuffer<float> L2_in(L2_in_py);
        DeviceBuffer<float> weight(weight_py);
        DeviceBuffer<float> L3_grad(L3_grad_py);

        DeviceBuffer<float> L1_grad(L1_grad_py.size());
        DeviceBuffer<float> L2_grad(L2_grad_py.size());
        DeviceBuffer<float> weight_grad(weight_grad_py.size());

        backward(L3_grad_host.shape[0], 
                L1_in.ptr, L1_grad.ptr,
                L2_in.ptr, L2_grad.ptr,
                weight.ptr, weight_grad.ptr,
                L3_grad.ptr);

        L1_grad.copy_to_host_buffer(L1_grad_host);
        L2_grad.copy_to_host_buffer(L2_grad_host);
        weight_grad.copy_to_host_buffer(weight_grad_host);
    }

    /*
    * This benchmarking function does not clear cache, etc. between runs. It copies
    * data from the CPU to the GPU, but only once. This time is not included in benchmarking.
    */
    void benchmark_forward_cpu(
            py::array_t<float> L1_in_py,
            py::array_t<float> L2_in_py,
            py::array_t<float> L3_out_py,
            py::array_t<float> weights_py,
            uint64_t num_warmup,
            py::array_t<float> time_millis_py);

    void benchmark_backward_cpu(
            py::array_t<float> L1_in_py, py::array_t<float> L1_grad_py,
            py::array_t<float> L2_in_py, py::array_t<float> L2_grad_py,
            py::array_t<float> weight_py, py::array_t<float> weight_grad_py,
            py::array_t<float> L3_grad_py,
            uint64_t num_warmup,
            py::array_t<float> time_millis_py);

    virtual ~GenericTensorProductImpl() {};
};

//=========================================================================

/*
* A simple implementation that gets each thread 
* to handle each tensor product based on a coordinate format. 
*/
class __attribute__ ((visibility ("default"))) ThreadTensorProductImpl : public GenericTensorProductImpl {
public:
    DeviceBuffer<uint8_t> coord1; 
    DeviceBuffer<uint8_t> coord2; 
    DeviceBuffer<uint8_t> coord3; 
    DeviceBuffer<float> values;

    ThreadTensorProductImpl(
        RepTriple &reps,
        py::array_t<uint8_t> coord1_py, 
        py::array_t<uint8_t> coord2_py,
        py::array_t<uint8_t> coord3_py,
        py::array_t<float> values_py 
        ) :
        GenericTensorProductImpl(reps),
        coord1(coord1_py),
        coord2(coord2_py),
        coord3(coord3_py),
        values(values_py)
        { 
            if(L1.irreps.size() != 1 || L2.irreps.size() != 1 || L3.irreps.size() != 1) {
                throw std::invalid_argument("ThreadTensorProductImpl only supports single irreps");
            }
        }

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~ThreadTensorProductImpl() = default;
};


//=========================================================================
/*
* A tensor product that executes a dense GEMM after instantiating Kronecker 
* products explicitly using cuBLASLt. 
*/
class __attribute__ ((visibility ("default"))) GemmTensorProductImpl : public GenericTensorProductImpl {
public:
    size_t workspaceSize = 1024 * 1024 * 4;
    DeviceBuffer<char> workspace;

    uint64_t num_products;
    DeviceBuffer<float> cg_coeffs;
    DeviceBuffer<float> kprods;

    cublasLtHandle_t     ltHandle;
    cublasLtMatmulDesc_t operationDesc = NULL; 
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL; 
    cublasLtMatmulPreference_t preference = NULL; 
    cublasLtMatmulHeuristicResult_t heuristicResult {};

    GemmTensorProductImpl(
        RepTriple &reps,
        uint64_t num_products1,
        py::array_t<float> cg_coeffs_py 
        ) :
        GenericTensorProductImpl(reps),
        workspace(workspaceSize),
        num_products(num_products1),
        cg_coeffs(cg_coeffs_py), 
        kprods(num_products * L1.get_rep_length() * L2.get_rep_length())
        { preprocess(); }

    void preprocess();

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~GemmTensorProductImpl(); 
};


//=========================================================================
/*
* A tensor product that uses shuffle primitives. Each tensor product is 
* assigned to a single warp. 
*/
class __attribute__ ((visibility ("default"))) ShuffleTensorProductImpl : public GenericTensorProductImpl {
public:
    int max_lane_length, reduction_depth;

    DeviceBuffer<float> warp_values;
    DeviceBuffer<int> l1_indices;
    DeviceBuffer<int> l2_indices;
    DeviceBuffer<int> red_lanes;

    JITKernel jit;

    ShuffleTensorProductImpl(
        RepTriple &reps,
        py::array_t<float> warp_values_py, 
        py::array_t<int> l1_indices_py, 
        py::array_t<int> l2_indices_py, 
        py::array_t<int> red_lanes_py);

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    ~ShuffleTensorProductImpl() = default; 
};


//=========================================================================
/*
* A tensor product where we write out all instructions into a JIT-compiled kernel.
*/
class __attribute__ ((visibility ("default"))) UnrollTPImpl : public GenericTensorProductImpl {
public:
    JITKernel jit;
    KernelLaunchConfig &forward_config; 
    KernelLaunchConfig &backward_config; 

    UnrollTPImpl(
        RepTriple &reps,
        std::string jit_kernel,    
        KernelLaunchConfig &forward_config_i,  
        KernelLaunchConfig &backward_config_i);

    void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights);

    void backward(
            uint64_t num_products,
            float* L1_in, float* L1_grad,
            float* L2_in, float* L2_grad,
            float* weight, float* weight_grad,
            float* L3_grad); 

    ~UnrollTPImpl() = default; 
};

