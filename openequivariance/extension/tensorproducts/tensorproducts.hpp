#pragma once

#include <stdexcept>
#include <cstdint>
#include <string>

#include "jit.hpp"

class __attribute__ ((visibility ("default"))) GenericTensorProductImpl {
public:
    GenericTensorProductImpl() { }

    virtual void exec_tensor_product(uint64_t num_products,
            void* L1_in, void* L2_in, void* L3_out, void* weights) = 0;

    void exec_tensor_product_device_rawptrs(uint64_t num_products,
            uint64_t L1_in, uint64_t L2_in, uint64_t L3_out, uint64_t weights) {
        
        exec_tensor_product(num_products,
            reinterpret_cast<void*>(L1_in),
            reinterpret_cast<void*>(L2_in),
            reinterpret_cast<void*>(L3_out),
            reinterpret_cast<void*>(weights));
    } 

    virtual void backward(size_t num_products,
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad) {

        throw std::logic_error("Backward pass not implemented yet!");
    }

    void backward_device_rawptrs(uint64_t num_products,
            uint64_t L1_in, uint64_t L1_grad,
            uint64_t L2_in, uint64_t L2_grad, 
            uint64_t weight, uint64_t weight_grad,
            uint64_t L3_grad) {

        backward(num_products,
            reinterpret_cast<void*>(L1_in), reinterpret_cast<void*>(L1_grad),
            reinterpret_cast<void*>(L2_in), reinterpret_cast<void*>(L2_grad),
            reinterpret_cast<void*>(weight), reinterpret_cast<void*>(weight_grad),
            reinterpret_cast<void*>(L3_grad)
        );
    }

    virtual ~GenericTensorProductImpl() {};
};

class __attribute__ ((visibility ("default"))) JITTPImpl : public GenericTensorProductImpl {
public:
    JITKernel jit;
    KernelLaunchConfig forward_config; 
    KernelLaunchConfig backward_config; 

    JITTPImpl(std::string jit_kernel,    
        KernelLaunchConfig &forward_config_i,  
        KernelLaunchConfig &backward_config_i);

    void exec_tensor_product(uint64_t num_products,
            void* L1_in, void* L2_in, void* L3_out, void* weights);

    void backward(uint64_t num_products,
            void* L1_in, void* L1_grad,
            void* L2_in, void* L2_grad,
            void* weight, void* weight_grad,
            void* L3_grad); 

    ~JITTPImpl() = default; 
};