#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "buffer.hpp"

using namespace std;
namespace py = pybind11;

// Taken from Stack Overflow
inline size_t round_up(size_t in, size_t multiple) {
    if (multiple == 0)
        return in;

    int remainder = in % multiple;
    if (remainder == 0)
        return in ;

    return in + multiple - remainder;
}

/*
* Graph convolution that combines node / edge features 
*/

class ESPMM_Context {  
public:
    // TODO: have this work for a sum of reps, not just one. 
    uint64_t node_count;
    uint64_t L1; // X_in representation
    uint64_t L2; // Edge feature representation
    uint64_t L3; // X_out representation

    size_t X_in_rowlen;
    size_t edge_rowlen;
    size_t X_out_rowlen; 

    ESPMM_Context(
        uint64_t node_count_i,
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i) :
        node_count(node_count_i),
        L1(L1_i), L2(L2_i), L3(L3_i),
        X_in_rowlen(round_up(L1 * 2 + 1, 128 / sizeof(float))),
        edge_rowlen(round_up(L2 * 2 + 1, 128 / sizeof(float))),
        X_out_rowlen(round_up(L3 * 2 + 1, 128 / sizeof(float)))
        { }

    size_t get_X_in_rowlen() {
        return X_in_rowlen;
    }

    size_t get_edge_rowlen() {
        return edge_rowlen;
    }

    size_t get_X_out_rowlen() {
        return X_out_rowlen;
    }
};

class __attribute__ ((visibility ("default"))) GenericTensorProductImpl {
public:
    uint64_t L1; // X_in representation
    uint64_t L2; // Edge feature representation
    uint64_t L3; // X_out representation

    size_t L1_rowlen;
    size_t L2_rowlen;
    size_t L3_rowlen;

    GenericTensorProductImpl(
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i) :
        L1(L1_i), L2(L2_i), L3(L3_i),
        L1_rowlen(L1 * 2 + 1),
        L2_rowlen(L2 * 2 + 1),
        L3_rowlen(L3 * 2 + 1)
        { }

    size_t get_row_length(int mode) {
        switch(mode) {
            case 1:
                return L1_rowlen;
            case 2:
                return L2_rowlen;
            case 3:
                return L3_rowlen;
            default:
                throw std::invalid_argument( "Invalid mode!" );

        }
    }

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

    virtual ~GenericTensorProductImpl() {};
};

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

    /*L1_rowlen(round_up(L1 * 2 + 1, 128 / sizeof(float))),
    L2_rowlen(round_up(L2 * 2 + 1, 128 / sizeof(float))),
    L3_rowlen(round_up(L3 * 2 + 1, 128 / sizeof(float)))*/

    ThreadTensorProductImpl(
        uint64_t L1_i, 
        uint64_t L2_i, 
        uint64_t L3_i,
        py::array_t<uint8_t> coord1_py, 
        py::array_t<uint8_t> coord2_py,
        py::array_t<uint8_t> coord3_py,
        py::array_t<float> values_py 
        ) :
        GenericTensorProductImpl(L1_i, L2_i, L3_i),
        coord1(coord1_py),
        coord2(coord2_py),
        coord3(coord3_py),
        values(values_py)
        { }

    void exec_tensor_product(
            uint64_t num_products,
            float* X_in,
            float* X_out,
            float* edge_features);

    ~ThreadTensorProductImpl() = default;
};

