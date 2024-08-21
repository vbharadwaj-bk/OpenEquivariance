//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"
#include "utility.hpp"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<GenericTensorProductImpl>(m, "GenericTensorProductImpl")
        .def("get_row_length", &GenericTensorProductImpl::get_row_length)
        .def("exec_tensor_product_cpu", &GenericTensorProductImpl::exec_tensor_product_cpu);
    py::class_<ThreadTensorProductImpl, GenericTensorProductImpl>(m, "ThreadTensorProductImpl")
        .def(py::init<uint64_t, uint64_t, uint64_t>());
}

/*
<%
setup_pybind11(cfg)

import os
cwd = os.getcwd()
print(cwd)

espmm_path = f'{cwd}/build/lib'
rpath_options = f'-Wl,-rpath,{espmm_path}'

compile_args = [f'-I{cwd}/build/include']
link_args = [f'-L{espmm_path}', rpath_options, '-lespmm']

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")

cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args
cfg['dependencies'] = [f'{cwd}/build/include/espmm.hpp']
%>
*/
