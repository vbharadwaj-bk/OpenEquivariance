#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "buffer.hpp"
#include "jit.hpp"

class __attribute__ ((visibility ("default"))) ConvolutionImpl {
public:
    bool record_internal_stats = false;

    ConvolutionImpl() {
    }

    virtual void exec_conv(
        void* L1_in,
        void* L2_in,
        void* weights, 
        void* L3_out,
        void* rows,
        void* cols,
        uint64_t nnz,
        uint64_t node_count,
        bool disable_tensor_op) = 0;

    void exec_conv_rawptrs(
        uint64_t L1_in,
        uint64_t L2_in,
        uint64_t weights,
        uint64_t L3_out,
        uint64_t rows,
        uint64_t cols,
        uint64_t nnz,
        uint64_t node_count,
        bool disable_tensor_op) {

        exec_conv(
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(weights),
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(rows),
            reinterpret_cast<void*>(cols),
            nnz,
            node_count,
            disable_tensor_op);
    }

    virtual void backward(
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad,
        void* rows, void* cols,
        uint64_t nnz, uint64_t node_count,
        bool disable_tensor_op) = 0;

    void backward_rawptrs(
        uint64_t L1_in, uint64_t L1_grad,
        uint64_t L2_in, uint64_t L2_grad,
        uint64_t weight, uint64_t weight_grad,
        uint64_t L3_grad,
        uint64_t rows, uint64_t cols,
        uint64_t nnz, uint64_t node_count,
        bool disable_tensor_op) {

        backward(
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight),
            reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad),
            reinterpret_cast<void*>(rows),
            reinterpret_cast<void*>(cols),
            nnz,
            node_count,
            disable_tensor_op);
    }

    virtual ~ConvolutionImpl() {};
};


class __attribute__ ((visibility ("default"))) JITConvImpl : public ConvolutionImpl{
public:
    JITKernel jit;
    KernelLaunchConfig forward_config; 
    KernelLaunchConfig backward_config; 

    JITConvImpl(
        std::string jit_kernel,    
        KernelLaunchConfig &forward_config_i,  
        KernelLaunchConfig &backward_config_i);

    void exec_conv(
        void* L1_in,
        void* L2_in,
        void* weights, 
        void* L3_out,
        void* rows,
        void* cols,
        uint64_t nnz,
        uint64_t node_count,
        bool disable_tensor_op); 

    void backward(
        void* L1_in, void* L1_grad,
        void* L2_in, void* L2_grad,
        void* weight, void* weight_grad,
        void* L3_grad,
        void* rows, void* cols,
        uint64_t nnz, uint64_t node_count,
        bool disable_tensor_op);

    ~JITConvImpl() = default; 
};