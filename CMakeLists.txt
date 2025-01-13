cmake_minimum_required(VERSION 3.15)
project(equivariant_spmm LANGUAGES CXX)

find_package(CUDAToolkit)

add_subdirectory(src/kernel)
add_subdirectory(src/wrapper)
