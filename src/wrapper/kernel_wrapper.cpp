#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(kernel_wrapper, m) {
    //========== Generic Utilities ============
    py::class_<Representation>(m, "Representation")
        .def(py::init<string>())
        .def(py::init<int, int>())
        .def(py::init<int>())
        .def("to_string", &Representation::to_string)
        .def("get_rep_length", &Representation::get_rep_length)
        .def("num_irreps", &Representation::num_irreps)
        .def("mult", &Representation::mult)
        .def("type", &Representation::type)
        .def("even", &Representation::even) 
        .def("get_irrep_offsets", &Representation::get_irrep_offsets) 
        .def("transpose_irreps_cpu", &Representation::transpose_irreps_cpu);
    py::class_<KernelLaunchConfig>(m, "KernelLaunchConfig")
        .def(py::init<>())
        .def_readwrite("num_blocks", &KernelLaunchConfig::num_blocks)
        .def_readwrite("warp_size", &KernelLaunchConfig::warp_size)
        .def_readwrite("num_threads", &KernelLaunchConfig::num_threads)
        .def_readwrite("smem", &KernelLaunchConfig::smem);

    //=========== Batch tensor products =========
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("exec_tensor_product", &GenericTensorProductImpl::exec_tensor_product_device_rawptrs)
        .def("backward", &GenericTensorProductImpl::backward_device_rawptrs);
    py::class_<JITTPImpl, GenericTensorProductImpl>(m, "JITTPImpl")
        .def(py::init<std::string, KernelLaunchConfig&, KernelLaunchConfig&>());

    //============= Convolutions ===============
    py::class_<ConvolutionImpl>(m, "ConvolutionImpl")
        .def("exec_conv_rawptrs", &ConvolutionImpl::exec_conv_rawptrs)
        .def("backward_rawptrs", &ConvolutionImpl::backward_rawptrs);
    py::class_<JITConvImpl, ConvolutionImpl>(m, "JITConvImpl")
        .def(py::init<std::string, KernelLaunchConfig&, KernelLaunchConfig&>());
    py::class_<DeviceProp>(m, "DeviceProp")
        .def(py::init<int>())
        .def_readonly("warpsize", &DeviceProp::warpsize)
        .def_readonly("major", &DeviceProp::major)
        .def_readonly("minor", &DeviceProp::minor)
        .def_readonly("multiprocessorCount", &DeviceProp::multiprocessorCount)
        .def_readonly("maxSharedMemPerBlock", &DeviceProp::maxSharedMemPerBlock); 
    py::class_<PyDeviceBuffer>(m, "DeviceBuffer")
        .def(py::init<py::buffer>())
        .def("copy_to_host", &PyDeviceBuffer::copy_to_host)
        .def("data_ptr", &PyDeviceBuffer::data_ptr);
    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop_clock_get_elapsed", &GPUTimer::stop_clock_get_elapsed);
}