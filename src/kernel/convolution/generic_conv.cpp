#include "convolution.hpp"
#include "gpu_util.hpp"
#include "buffer.hpp"

using namespace std;

void ConvolutionImpl::benchmark_cpu(
        py::array_t<float> &L1_in_py,
        py::array_t<float> &L2_in_py,
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
    DeviceBuffer<float> L3_out(L3_out_host.size());

    DeviceBuffer<float> coords(coords_py); 
    DeviceBuffer<uint32_t> rows(rows_py); 
    DeviceBuffer<uint32_t> cols(cols_py);

    uint64_t nnz = rows_host.shape[0];
    uint32_t node_count = static_cast<uint32_t>(L3_out_host.shape[0]);

    record_internal_stats = false;

    for(int i = 0; i < num_warmup; i++) {
        exec_conv(L1_in.ptr, L2_in.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count, disable_tensor_op);
    }

    record_internal_stats = true;
    // TODO: Synchronization can be costly if the runtime of any given
    // kernel execution is small. 
    for(int i = 0; i < time_millis.shape[0]; i++) {
        timer.start();
        exec_conv(L1_in.ptr, L2_in.ptr, L3_out.ptr, rows.ptr, cols.ptr, nnz, node_count, disable_tensor_op);
        float elapsed = timer.stop_clock_get_elapsed();
        time_millis[i] = elapsed;
    }

    record_internal_stats = false;
    L3_out.copy_to_host_buffer(L3_out_host);
}
