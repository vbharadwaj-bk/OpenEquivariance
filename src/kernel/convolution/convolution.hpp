#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "buffer.hpp"
#include "representation.hpp"
#include "jit.hpp"

class __attribute__ ((visibility ("default"))) ConvolutionImpl {
public:
    Representation &L1;
    Representation &L2;
    Representation &L3;

    bool record_internal_stats = false;

    ConvolutionImpl(RepTriple &io_reps) :
        L1(io_reps.L1),
        L2(io_reps.L2),
        L3(io_reps.L3) { }

    virtual void exec_conv(
            float* L1_in,
            float* L2_in,
            float* L3_out,
            uint32_t* rows,
            uint32_t* cols,
            uint64_t nnz,
            uint32_t node_count,
            bool disable_tensor_op
            ) = 0; 

    void exec_conv_cpu(
            py::array_t<float> &L1_in_py,
            py::array_t<float> &L2_in_py,
            py::array_t<float> &L3_out_py,
            py::array_t<float> &coords_py,
            py::array_t<uint32_t> &rows_py,
            py::array_t<uint32_t> &cols_py,
            bool disable_tensor_op) {

        Buffer<float> L3_out_host(L3_out_py);
        Buffer<uint32_t> rows_host(rows_py);

        DeviceBuffer<float> L1_in(L1_in_py);
        DeviceBuffer<float> L2_in(L2_in_py);
        DeviceBuffer<float> L3_out(L3_out_host.size());

        // Transfer rows, cols, and coords to device. 
        DeviceBuffer<float> coords(coords_py); 
        DeviceBuffer<uint32_t> rows(rows_py); 
        DeviceBuffer<uint32_t> cols(cols_py);

        uint64_t nnz = rows_host.shape[0];
        uint32_t node_count = static_cast<uint32_t>(L3_out_host.shape[0]);

        exec_conv(L1_in.ptr, L2_in.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count, disable_tensor_op);
        L3_out.copy_to_host_buffer(L3_out_host);
    }

    void benchmark_cpu(
            py::array_t<float> &L1_in_py,
            py::array_t<float> &L2_in_py,
            py::array_t<float> &L3_out_py,
            py::array_t<float> &coords_py,
            py::array_t<uint32_t> &rows_py,
            py::array_t<uint32_t> &cols_py,
            bool disable_tensor_op,
            uint64_t num_warmup,
            py::array_t<float> time_millis_py);

    virtual ~ConvolutionImpl() {};
};

//=========================================================================
/*
* Simple implementation that assigns one warp per nonzero and
* executes atomicAdd operations to accumulate to the output buffer.
*/
class __attribute__ ((visibility ("default"))) AtomicConvImpl  : public ConvolutionImpl {
public:
    AtomicConvImpl(RepTriple &io_reps) :
        ConvolutionImpl(io_reps) { };

    void exec_conv(
            float* L1_in,
            float* L2_in,
            float* L3_out,
            uint32_t* rows,
            uint32_t* cols,
            uint64_t nnz,
            uint32_t node_count,
            bool disable_tensor_op
            );

    ~AtomicConvImpl() = default;
};

