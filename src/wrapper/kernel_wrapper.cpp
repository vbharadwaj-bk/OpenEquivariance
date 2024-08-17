//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"
#include "utility.hpp"

using namespace std;
namespace py = pybind11;

void equivariant_spmm_cpu_wrapped(
        ESPMM_Context &context, 
        py::array_t<uint64_t> rows_py,
        py::array_t<uint64_t> cols_py,
        py::array_t<float> X_in_py,
        py::array_t<float> edge_features_py,
        py::array_t<float> X_out_py) {

    Buffer<uint64_t> rows(rows_py);
    Buffer<uint64_t> cols(cols_py);
    Buffer<float> X_in(X_in_py);
    Buffer<float> edge_features(edge_features_py);
    Buffer<float> X_out(X_out_py);

    equivariant_spmm_cpu(
        context,
        cols.shape[0],
        rows.ptr,
        cols.ptr,
        X_in.ptr,
        X_out.ptr,
        edge_features.ptr);
}

void exec_tensor_product_cpu_wrapped(
        TensorProduct &context, 
        py::array_t<float> L1_in_py,
        py::array_t<float> L2_in_py,
        py::array_t<float> L3_out_py) {

    Buffer<float> L1_in(L1_in_py);
    Buffer<float> L2_in(L2_in_py);
    Buffer<float> L3_out(L3_out_py);

    exec_tensor_product_cpu(
        context,
        L1_in.shape[0],
        L1_in.ptr,
        L2_in.ptr,
        L3_out.ptr);
}


PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<ESPMM_Context>(m, "ESPMM_Context")
        .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t>())
        .def("get_X_in_rowlen", &ESPMM_Context::get_X_in_rowlen)
        .def("get_edge_rowlen", &ESPMM_Context::get_edge_rowlen)
        .def("get_X_out_rowlen", &ESPMM_Context::get_X_out_rowlen);

    py::class_<TensorProduct>(m, "TensorProduct")
        .def(py::init<uint64_t, uint64_t, uint64_t>())
        .def("get_L1_rowlen", &TensorProduct::get_L1_rowlen)
        .def("get_L2_rowlen", &TensorProduct::get_L2_rowlen)
        .def("get_L3_rowlen", &TensorProduct::get_L3_rowlen);

    m.def("equivariant_spmm_cpu", &equivariant_spmm_cpu_wrapped);
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
