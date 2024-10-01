cmake_minimum_required(VERSION 3.15)
project(equivariant_spmm LANGUAGES CXX CUDA)


# Set CUDA version
set(CUDA_VERSION 11.0)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit)
else()
  message(STATUS "No CUDA compiler found")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

find_package(OpenMP)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_80 -fPIC)

add_subdirectory(src/kernel)
add_subdirectory(src/wrapper)
