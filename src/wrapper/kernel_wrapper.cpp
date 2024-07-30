//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"
#include "utility.hpp"

using namespace std;
namespace py = pybind11;

void equivariant_spmm_cpu_wrapped( 
        uint64_t L1, 
        uint64_t L2,
        uint64_t L3,
        py::array_t<uint64_t> row_ptr_py,
        py::array_t<uint64_t> cols_py,
        py::array_t<double> X_in_py,
        py::array_t<double> X_out_py,
        py::array_t<double> edge_features_py) {

    Buffer<uint64_t> row_ptr(row_ptr_py);
    Buffer<uint64_t> cols(cols_py);
    Buffer<double> X_in(X_in_py);
    Buffer<double> X_out(X_out_py);
    Buffer<double> edge_features(edge_features_py);

    equivariant_spmm_cpu(
        X_in_py.shape[0],
        edge_features_py.shape[0],
        L1, L2, L3,
        row_ptr.ptr,
        cols.ptr,
        X_in.ptr,
        X_out.ptr,
        edge_features.ptr);
}

PYBIND11_MODULE(kernel_wrapper, m) {
    m.def("equivariant_spmm_cpu", &equivariant_spmm_wrapped);
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
