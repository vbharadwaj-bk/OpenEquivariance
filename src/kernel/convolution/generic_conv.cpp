#include "convolution.hpp"
#include "gpu_util.hpp"
#include "buffer.hpp"

using namespace std;

void ConvolutionImpl::benchmark_forward_cpu(
        py::array_t<float> &L1_in_py,
        py::array_t<float> &L2_in_py,
        py::array_t<float> &weights_py,
        py::array_t<float> &L3_out_py,
        py::array_t<float> &coords_py,
        py::array_t<uint32_t> &rows_py,
        py::array_t<uint32_t> &cols_py,
        bool disable_tensor_op,
        uint64_t num_warmup,
        py::array_t<float> time_millis_py) {

    GPUTimer timer;
    Buffer<float> time_millis(time_millis_py);

    Buffer<float> L3_out_host(L3_out_py);
    Buffer<float> rows_host(rows_py);

    DeviceBuffer<float> L1_in(L1_in_py);
    DeviceBuffer<float> L2_in(L2_in_py);
    DeviceBuffer<float> weights(weights_py);
    DeviceBuffer<float> L3_out(L3_out_host.size());

    DeviceBuffer<float> coords(coords_py); 
    DeviceBuffer<uint32_t> rows(rows_py); 
    DeviceBuffer<uint32_t> cols(cols_py);

    uint64_t nnz = rows_host.shape[0];
    uint32_t node_count = static_cast<uint32_t>(L3_out_host.shape[0]);

    record_internal_stats = false;

    for(int i = 0; i < num_warmup; i++) {
        exec_conv(L1_in.ptr, L2_in.ptr, weights.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count, disable_tensor_op);
    }

    record_internal_stats = true;
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        exec_conv(L1_in.ptr, L2_in.ptr, weights.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count, disable_tensor_op);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }
    record_internal_stats = false;

    L3_out.copy_to_host_buffer(L3_out_host);
}

void ConvolutionImpl::benchmark_backward_cpu(
        py::array_t<float> L1_in_py, py::array_t<float> L1_grad_py,
        py::array_t<float> L2_in_py, py::array_t<float> L2_grad_py,
        py::array_t<float> weight_py, py::array_t<float> weight_grad_py,
        py::array_t<float> L3_grad_py,
        py::array_t<uint32_t> &rows_py,
        py::array_t<uint32_t> &cols_py,
        bool disable_tensor_op,
        uint64_t num_warmup,
        py::array_t<float> time_millis_py) {

    GPUTimer timer;
    Buffer<float> time_millis(time_millis_py);

    Buffer<float> L1_grad_host(L1_grad_py);
    Buffer<float> L2_grad_host(L2_grad_py);
    Buffer<float> L3_grad_host(L3_grad_py);
    Buffer<float> weight_grad_host(weight_grad_py);
    Buffer<uint32_t> rows_host(rows_py);

    // Copies data to device 
    DeviceBuffer<float> L1_in(L1_in_py);
    DeviceBuffer<float> L2_in(L2_in_py);
    DeviceBuffer<float> weight(weight_py);
    DeviceBuffer<float> L3_grad(L3_grad_py);

    DeviceBuffer<float> L1_grad(L1_grad_py.size());
    DeviceBuffer<float> L2_grad(L2_grad_py.size());
    DeviceBuffer<float> weight_grad(weight_grad_py.size());

    DeviceBuffer<uint32_t> rows(rows_py);
    DeviceBuffer<uint32_t> cols(cols_py);

    uint64_t nnz = rows_host.shape[0];
    uint32_t node_count = static_cast<uint32_t>(L3_grad_host.shape[0]);

    for(int i = 0; i < num_warmup; i++) {
        backward(L1_in.ptr, L1_grad.ptr,
                L2_in.ptr, L2_grad.ptr,
                weight.ptr, weight_grad.ptr,
                L3_grad.ptr,
                rows.ptr, cols.ptr, nnz, node_count,
                disable_tensor_op);
    }

    record_internal_stats = true;
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        backward(L1_in.ptr, L1_grad.ptr,
                L2_in.ptr, L2_grad.ptr,
                weight.ptr, weight_grad.ptr,
                L3_grad.ptr,
                rows.ptr, cols.ptr, nnz, node_count,
                disable_tensor_op);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }
    record_internal_stats = false;

    L1_grad.copy_to_host_buffer(L1_grad_host);
    L2_grad.copy_to_host_buffer(L2_grad_host);
    weight_grad.copy_to_host_buffer(weight_grad_host);
}
