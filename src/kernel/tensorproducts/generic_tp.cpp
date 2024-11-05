#include "tensorproducts.hpp"
#include "gpu_util.hpp"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

void GenericTensorProductImpl::benchmark_forward_cpu(
        py::array_t<float> L1_in_py,
        py::array_t<float> L2_in_py,
        py::array_t<float> weights_py, 
        py::array_t<float> L3_out_py,
        uint64_t num_warmup,
        py::array_t<float> time_millis_py) {

    GPUTimer timer;

    Buffer<float> time_millis(time_millis_py);
    Buffer<float> L3_out_host(L3_out_py);

    DeviceBuffer<float> L1_in(L1_in_py);
    DeviceBuffer<float> L2_in(L2_in_py);
    DeviceBuffer<float> weights(weights_py);
    DeviceBuffer<float> L3_out(L3_out_host.size());

    record_internal_stats = false;

    for(int i = 0; i < num_warmup; i++) {
        exec_tensor_product(L3_out_host.shape[0], L1_in.ptr, L2_in.ptr, weights.ptr, L3_out.ptr);
    }

    record_internal_stats = true;
    // TODO: Synchronization can be costly if the runtime of any given
    // kernel execution is small. 
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        exec_tensor_product(L3_out_host.shape[0], L1_in.ptr, L2_in.ptr, weights.ptr, L3_out.ptr);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }

    record_internal_stats = false;
    L3_out.copy_to_host_buffer(L3_out_host);
}


void GenericTensorProductImpl::benchmark_backward_cpu(
        py::array_t<float> L1_in_py, py::array_t<float> L1_grad_py,
        py::array_t<float> L2_in_py, py::array_t<float> L2_grad_py,
        py::array_t<float> weight_py, py::array_t<float> weight_grad_py,
        py::array_t<float> L3_grad_py,
        uint64_t num_warmup,
        py::array_t<float> time_millis_py) {

    GPUTimer timer;
    Buffer<float> time_millis(time_millis_py);

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

    record_internal_stats = false;
    for(int i = 0; i < num_warmup; i++) {
        backward(L3_grad_host.shape[0], 
                L1_in.ptr, L1_grad.ptr,
                L2_in.ptr, L2_grad.ptr,
                weight.ptr, weight_grad.ptr,
                L3_grad.ptr);
    }

    record_internal_stats = true;
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        backward(L3_grad_host.shape[0], 
                L1_in.ptr, L1_grad.ptr,
                L2_in.ptr, L2_grad.ptr,
                weight.ptr, weight_grad.ptr,
                L3_grad.ptr);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }
    record_internal_stats = false;

    L1_grad.copy_to_host_buffer(L1_grad_host);
    L2_grad.copy_to_host_buffer(L2_grad_host);
    weight_grad.copy_to_host_buffer(weight_grad_host);
}
