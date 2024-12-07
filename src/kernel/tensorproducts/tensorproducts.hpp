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

