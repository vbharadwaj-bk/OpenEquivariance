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

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture=sm_80 -fPIC)

# If not set, set variables for the Python3 interpreter and python3-config
# default values should be python3 and python3-config, respectively
#if(NOT DEFINED PYTHON_EXECUTABLE)
#  set(PYTHON_EXECUTABLE "python3")
#endif()

#if(NOT DEFINED PYTHON_CONFIG_EXECUTABLE)
#  set(PYTHON_CONFIG_EXECUTABLE "python3-config")
#endif()

#execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --includes
#                OUTPUT_VARIABLE PYTHON_INCLUDES)
#execute_process(COMMAND ${PYTHON_CONFIG_EXECUTABLE} --extension-suffix 
#                OUTPUT_VARIABLE EXTENSION_SUFFIX)

# 

add_subdirectory(src/kernel)
add_subdirectory(src/wrapper)
