cmake_minimum_required(VERSION 3.10)
project(equivariant_spmm LANGUAGES CXX CUDA)

# Set CUDA version
set(CUDA_VERSION 11.0)

# Find CUDA
find_package(CUDA ${CUDA_VERSION} REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_80)

add_subdirectory(src/kernel)
