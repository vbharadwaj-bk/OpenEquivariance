#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("exec_tensor_product", &GenericTensorProductImpl::exec_tensor_product)
        .def("exec_tensor_product_cpu", &GenericTensorProductImpl::exec_tensor_product_cpu)
        .def("benchmark_cpu", &GenericTensorProductImpl::benchmark_cpu);
    py::class_<ThreadTensorProductImpl, GenericTensorProductImpl>(m, "ThreadTensorProductImpl")
        .def(py::init<Representation&, Representation&, Representation&,
            py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<float>>());
    py::class_<GemmTensorProductImpl, GenericTensorProductImpl>(m, "GemmTensorProductImpl")
        .def(py::init<uint64_t, Representation&, Representation&, Representation&, py::array_t<float>>());
    py::class_<ShuffleTensorProductImpl, GenericTensorProductImpl>(m, "ShuffleTensorProductImpl")
        .def(py::init<Representation&, Representation&, Representation&, 
                py::array_t<float>, py::array_t<int>, py::array_t<int>, py::array_t<int>>());
    py::class_<UnrollTPImpl, GenericTensorProductImpl>(m, "UnrollTPImpl")
        .def(py::init<Representation&, Representation&, Representation&, std::string, KernelLaunchConfig&>());
    py::class_<Representation>(m, "Representation")
        .def(py::init<string>())
        .def(py::init<int, int>())
        .def(py::init<int>())
        .def("to_string", &Representation::to_string)
        .def("get_rep_length", &Representation::get_rep_length)
        .def("num_irreps", &Representation::num_irreps)
        .def("mult", &Representation::mult)
        .def("type", &Representation::type)
        .def("even", &Representation::even);
    py::class_<RepTriple>(m, "RepTriple")
        .def(py::init<Representation&, Representation&, Representation&>())
        .def_readwrite("L1", &RepTriple::L1) 
        .def_readwrite("L2", &RepTriple::L2)
        .def_readwrite("L3", &RepTriple::L3);
    py::class_<KernelLaunchConfig>(m, "KernelLaunchConfig")
        .def(py::init<>())
        .def_readwrite("num_blocks", &KernelLaunchConfig::num_blocks)
        .def_readwrite("num_threads", &KernelLaunchConfig::num_threads);
}