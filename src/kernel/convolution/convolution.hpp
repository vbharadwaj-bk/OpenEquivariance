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
    bool record_internal_stats = false;

    ConvolutionImpl() {
    }

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


class __attribute__ ((visibility ("default"))) JITConvImpl : public ConvolutionImpl{
public:
    JITKernel jit;
    KernelLaunchConfig &forward_config; 
    KernelLaunchConfig &backward_config; 

    JITConvImpl(
        std::string jit_kernel,    
        KernelLaunchConfig &forward_config_i,  
        KernelLaunchConfig &backward_config_i);

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

    ~JITConvImpl() = default; 
};


