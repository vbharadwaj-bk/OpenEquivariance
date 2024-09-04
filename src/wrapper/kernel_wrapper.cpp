//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("get_row_length", &GenericTensorProductImpl::get_row_length)
        .def("exec_tensor_product", &GenericTensorProductImpl::exec_tensor_product)
        .def("exec_tensor_product_cpu", &GenericTensorProductImpl::exec_tensor_product_cpu)
        .def("benchmark_cpu", &GenericTensorProductImpl::benchmark_cpu);
    py::class_<ThreadTensorProductImpl, GenericTensorProductImpl>(m, "ThreadTensorProductImpl")
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
            py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<uint8_t>, py::array_t<float>>());
    py::class_<GemmTensorProductImpl, GenericTensorProductImpl>(m, "GemmTensorProductImpl")
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t, py::array_t<float>>());
    py::class_<Representation>(m, "Representation")
        .def(py::init<string>())
        .def(py::init<int, int>())
        .def(py::init<int>())
        .def("to_string", &Representation::to_string)
        .def("get_rep_length", &Representation::get_rep_length);
}

/*
<%
setup_pybind11(cfg)

import os
cwd = os.getcwd()
headers = os.listdir(f'{cwd}/build/include')

espmm_path = f'{cwd}/build/lib'
rpath_options = f'-Wl,-rpath,{espmm_path}'

compile_args = [f'-I{cwd}/build/include']
link_args = [f'-L{espmm_path}', rpath_options, '-lespmm']

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")

cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args
cfg['dependencies'] = [f'{cwd}/build/include/{header}'
    for header in headers
]
%>
*/
