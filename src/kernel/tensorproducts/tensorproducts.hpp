#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cublasLt.h>

#include "buffer.hpp"
#include "representation.hpp"

class __attribute__ ((visibility ("default"))) GenericTensorProductImpl {
public:
    Representation &L1;
    Representation &L2;
    Representation &L3;

    bool record_internal_stats = false; 

    GenericTensorProductImpl(
        Representation &L1_i,
        Representation &L2_i,
        Representation &L3_i) :
        L1(L1_i), L2(L2_i), L3(L3_i)
        { }

    virtual void exec_tensor_product(
            uint64_t num_products,
            float* L1_in,
            float* L2_in,
            float* L3_out) = 0;

    // Executes function with CPU inputs from Python. Issues
    // memcpy to / from device. 
    void exec_tensor_product_cpu(
            py::array_t<float> L1_in_py,
            py::array_t<float> L2_in_py,
            py::array_t<float> L3_out_py) {
        
        // To get batch dimension 
        Buffer<float> L3_out_host(L3_out_py);

        // Copies data to device 
        DeviceBuffer<float> L1_in(L1_in_py);
        DeviceBuffer<float> L2_in(L2_in_py);
        DeviceBuffer<float> L3_out(L3_out_host.size());

        exec_tensor_product(L3_out_host.shape[0], L1_in.ptr, L2_in.ptr, L3_out.ptr);
        L3_out.copy_to_host_buffer(L3_out_host);
    }

    /*
    * This benchmarking function does not clear cache, etc. between runs. It copies
    * data from the CPU to the GPU, but only once. This time is not included in benchmarking.
    */
    void benchmark_cpu(
            py::array_t<float> L1_in_py,
            py::array_t<float> L2_in_py,
            py::array_t<float> L3_out_py,
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
        Representation &L1,
        Representation &L2,
        Representation &L3,
        py::array_t<uint8_t> coord1_py, 
        py::array_t<uint8_t> coord2_py,
        py::array_t<uint8_t> coord3_py,
        py::array_t<float> values_py 
        ) :
        GenericTensorProductImpl(L1, L2, L3),
        coord1(coord1_py),
        coord2(coord2_py),
        coord3(coord3_py),
        values(values_py)
        { 
            if(L1.irreps.size() != 1 || L2.irreps.size() != 1 || L3.irreps.size() != 1) {
                throw std::invalid_argument("ThreadTensorProductImpl only supports single irreps");
            }
            // else if(L1.mult(0) != 1 || L2.mult(0) != 1 || L3.mult(0) != 1) {
            //     throw std::invalid_argument("ThreadTensorProductImpl only supports multiplicity 1");
            // }
        }

    void exec_tensor_product(
            uint64_t num_products,
            float* X_in,
            float* X_out,
            float* edge_features);

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
        uint64_t num_products1,
        Representation &L1_i,
        Representation &L2_i,
        Representation &L3_i,
        py::array_t<float> cg_coeffs_py 
        ) :
        GenericTensorProductImpl(L1_i, L2_i, L3_i),
        workspace(workspaceSize),
        num_products(num_products1),
        cg_coeffs(cg_coeffs_py), 
        kprods(num_products * L1.get_rep_length() * L2.get_rep_length())
        { preprocess(); }

    void preprocess();

    void exec_tensor_product(
            uint64_t num_products,
            float* X_in,
            float* X_out,
            float* edge_features);

    ~GemmTensorProductImpl(); 
};