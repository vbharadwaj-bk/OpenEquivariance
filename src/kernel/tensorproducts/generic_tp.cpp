#include "tensorproducts.hpp"
#include "gpu_util.hpp"

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

void GenericTensorProductImpl::benchmark_cpu(
        py::array_t<float> L1_in_py,
        py::array_t<float> L2_in_py,
        py::array_t<float> L3_out_py,
        uint64_t num_warmup,
        py::array_t<float> time_millis_py) {

    GPUTimer timer;

    Buffer<float> time_millis(time_millis_py);
    Buffer<float> L3_out_host(L3_out_py);

    DeviceBuffer<float> L1_in(L1_in_py);
    DeviceBuffer<float> L2_in(L2_in_py);
    DeviceBuffer<float> L3_out(L3_out_host.size());

    for(int i = 0; i < num_warmup; i++) {
        exec_tensor_product(L3_out_host.shape[0], L1_in.ptr, L2_in.ptr, L3_out.ptr);
    }

    // TODO: Synchronization can be costly if the runtime of any given
    // kernel execution is small. 
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        exec_tensor_product(L3_out_host.shape[0], L1_in.ptr, L2_in.ptr, L3_out.ptr);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }

    L3_out.copy_to_host_buffer(L3_out_host);
}
