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
    py::class_<RepTriple>(m, "RepTriple")
        .def(py::init<Representation&, Representation&, Representation&>())
        .def(py::init<Representation&, Representation&, int>())
        .def("to_string", &RepTriple::to_string)
        .def("num_interactions", &RepTriple::num_interactions)
        .def("interactions", &RepTriple::interactions)
        .def("num_trainable_weights", &RepTriple::num_trainable_weights)
        .def_readwrite("L1", &RepTriple::L1) 
        .def_readwrite("L2", &RepTriple::L2)
        .def_readwrite("L3", &RepTriple::L3);
    py::class_<KernelLaunchConfig>(m, "KernelLaunchConfig")
        .def(py::init<>())
        .def_readwrite("num_blocks", &KernelLaunchConfig::num_blocks)
        .def_readwrite("warp_size", &KernelLaunchConfig::warp_size)
        .def_readwrite("num_threads", &KernelLaunchConfig::num_threads)
        .def_readwrite("smem", &KernelLaunchConfig::smem);

    //=========== Batch tensor products =========
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("exec_tensor_product", &GenericTensorProductImpl::exec_tensor_product_device_rawptrs)
        .def("exec_tensor_product_cpu", &GenericTensorProductImpl::exec_tensor_product_cpu)
        .def("backward_cpu", &GenericTensorProductImpl::backward_cpu)
        .def("benchmark_forward_cpu", &GenericTensorProductImpl::benchmark_forward_cpu) 
        .def("benchmark_backward_cpu", &GenericTensorProductImpl::benchmark_backward_cpu);
    py::class_<ThreadTensorProductImpl, GenericTensorProductImpl>(m, "ThreadTensorProductImpl")
        .def(py::init<RepTriple&, 
            py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<float>>());
    py::class_<GemmTensorProductImpl, GenericTensorProductImpl>(m, "GemmTensorProductImpl")
        .def(py::init<RepTriple&, uint64_t, py::array_t<float>>());
    py::class_<ShuffleTensorProductImpl, GenericTensorProductImpl>(m, "ShuffleTensorProductImpl")
        .def(py::init<RepTriple&, py::array_t<float>, py::array_t<int>, py::array_t<int>, py::array_t<int>>());
    py::class_<UnrollTPImpl, GenericTensorProductImpl>(m, "UnrollTPImpl")
        .def(py::init<RepTriple&, std::string, KernelLaunchConfig&, KernelLaunchConfig&>());

    //============= Convolutions ===============
    py::class_<ConvolutionImpl>(m, "ConvolutionImpl")
        .def("exec_conv_cpu", &ConvolutionImpl::exec_conv_cpu)
        .def("benchmark_cpu", &ConvolutionImpl::benchmark_cpu);
    py::class_<AtomicConvImpl, ConvolutionImpl>(m, "AtomicConvImpl")
        .def(py::init<RepTriple&>()); 
    py::class_<SMConvImpl, ConvolutionImpl>(m, "SMConvImpl")
        .def(py::init<RepTriple&>()); 
}