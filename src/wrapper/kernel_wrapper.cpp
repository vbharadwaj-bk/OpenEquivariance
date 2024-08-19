//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"
#include "utility.hpp"

using namespace std;
namespace py = pybind11;

void exec_tensor_product_cpu_wrapped(
        TensorProduct &context, 
        py::array_t<float> L1_in_py,
        py::array_t<float> L2_in_py,
        py::array_t<float> L3_out_py) {

    Buffer<float> L1_in(L1_in_py);
    Buffer<float> L2_in(L2_in_py);
    Buffer<float> L3_out(L3_out_py);

    context.exec_tensor_product_cpu(
        L1_in.shape[0],
        L1_in.ptr,
        L2_in.ptr,
        L3_out.ptr);
}

PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<TensorProduct>(m, "TensorProduct")
        .def("get_row_length", &TensorProduct::get_row_length);

    py::class_<ThreadTensorProduct>(m, "TensorProduct")
        .def("get_row_length", &TensorProduct::get_row_length);

    m.def("exec_tensor_product_cpu", &exec_tensor_product_cpu_wrapped);
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
link_args = [f'-L{espmm_path}', '-lespmm', rpath_options]

print(f"Compiling C++ extensions with {compile_args}")
print(f"Linking C++ extensions with {link_args}")

cfg['extra_compile_args'] = compile_args 
cfg['extra_link_args'] = link_args
cfg['dependencies'] = [f'{cwd}/build/include/espmm.hpp']
%>
*/
