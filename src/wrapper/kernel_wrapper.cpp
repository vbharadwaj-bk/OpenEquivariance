//cppimport

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "espmm.hpp"

using namespace std;
namespace py = pybind11;


void equivariant_spmm_wrapped() {
    equivariant_spmm();
}

PYBIND11_MODULE(kernel_wrapper, m) {
    m.def("equivariant_spmm", &equivariant_spmm_wrapped);
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
