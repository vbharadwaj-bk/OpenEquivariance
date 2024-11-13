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
    bool record_internal_stats = false;

    GenericTensorProductImpl() { }

    virtual void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out,
            float* weights) = 0;

    void exec_tensor_product_device_rawptrs(
            uint64_t num_products,
            uint64_t L1_in,
            uint64_t L2_in,
            uint64_t L3_out,
            uint64_t weights) {
        
        exec_tensor_product(
            num_products,
            reinterpret_cast<float*>(L1_in),
            reinterpret_cast<float*>(L2_in),
            reinterpret_cast<float*>(L3_out),
            reinterpret_cast<float*>(weights));
    } 

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

    void backward_device_rawptrs(
            uint64_t num_products,
            uint64_t L1_in, uint64_t L1_grad,
            uint64_t L2_in, uint64_t L2_grad, 
            uint64_t weight, uint64_t weight_grad,
            uint64_t L3_grad) {

        backward(
            num_products,
            reinterpret_cast<float*>(L1_in), reinterpret_cast<float*>(L1_grad),
            reinterpret_cast<float*>(L2_in), reinterpret_cast<float*>(L2_grad),
            reinterpret_cast<float*>(weight), reinterpret_cast<float*>(weight_grad),
            reinterpret_cast<float*>(L3_grad)
        );
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
* A tensor product where we write out all instructions into a JIT-compiled kernel.
*/
class __attribute__ ((visibility ("default"))) JITTPImpl : public GenericTensorProductImpl {
public:
    JITKernel jit;
    KernelLaunchConfig &forward_config; 
    KernelLaunchConfig &backward_config; 

    JITTPImpl(
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

    ~JITTPImpl() = default; 
};

