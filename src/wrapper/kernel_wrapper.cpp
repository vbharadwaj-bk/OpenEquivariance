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
        py::array_t<double> X_in_py,
        py::array_t<double> edge_features_py,
        py::array_t<double> X_out_py) {

    Buffer<uint64_t> rows(rows_py);
    Buffer<uint64_t> cols(cols_py);
    Buffer<double> X_in(X_in_py);
    Buffer<double> edge_features(edge_features_py);
    Buffer<double> X_out(X_out_py);

    equivariant_spmm_cpu(
        context,
        edge_features.shape[0],
        rows.ptr,
        cols.ptr,
        X_in.ptr,
        X_out.ptr,
        edge_features.ptr);
}

PYBIND11_MODULE(kernel_wrapper, m) {
    py::class_<ESPMM_Context>(m, "ESPMM_Context")
    .def(py::init<uint64_t, uint64_t, uint64_t, uint64_t>());
    m.def("equivariant_spmm_cpu", &equivariant_spmm_cpu_wrapped);
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
