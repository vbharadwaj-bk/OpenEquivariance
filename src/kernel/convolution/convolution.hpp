#pragma once

#include <stdexcept>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>

#include "buffer.hpp"
#include "representation.hpp"
#include "jit.hpp"

class ConvolutionImpl {
public:
    Representation &L1;
    Representation &L2;
    Representation &L3;

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
            ) = 0; 

    void exec_conv_cpu(
            py::array_t<float> &L1_in_py,
            py::array_t<float> &L2_in_py,
            py::array_t<float> &L3_out_py,
            py::array_t<float> &coords_py,
            py::array_t<uint32_t> &rows_py,
            py::array_t<uint32_t> &cols_py) {

        Buffer<float> L3_out_host(L3_out_py);
        Buffer<uint32_t> rows_host(rows);

        DeviceBuffer<float> L1_in(L1_in_py);
        DeviceBuffer<float> L2_in(L2_in_py);
        DeviceBuffer<float> L3_out(L3_out_host.size());

        // Transfer rows, cols, and coords to graph.
        DeviceBuffer<float> coords(coords_py); 
        DeviceBuffer<uint32_t> rows(rows_py); 
        DeviceBuffer<uint32_t> cols(cols_py);

        uint64_t nnz = rows_host.shape[0];
        uint32_t node_count = static_cast<uint32_t>(L3_out.shape[0]);

        exec_conv(L1_in.ptr, L2_in.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count);
        L3_out.copy_to_host_buffer(L3_out_host);
    }

    virtual ~ConvolutionImpl() {};
};

