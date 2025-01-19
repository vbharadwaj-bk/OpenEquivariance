cmake_minimum_required(VERSION 3.15)
project(equivariant_spmm LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)

add_subdirectory(kernel)
add_subdirectory(wrapper)
