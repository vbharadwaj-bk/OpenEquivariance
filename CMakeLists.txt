cmake_minimum_required(VERSION 3.10)
project(equivariant_spmm LANGUAGES CXX CUDA)

# Helps find pybindcd
# This doesn't actually fix the problem
include_directories(/usr/include)

# Set CUDA version
set(CUDA_VERSION 11.0)

# Find CUDA
find_package(CUDA ${CUDA_VERSION} REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)
# This also doesn't fix the problem, but there might be a way to use pybind directly from cmake

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_80)

add_subdirectory(src/kernel)
