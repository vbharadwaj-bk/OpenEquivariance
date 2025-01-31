cmake_minimum_required(VERSION 3.15)
project(equivariant_spmm LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development)

set(core_SOURCES
  util/buffer.hpp
  util/jit.hpp
  util/backend_cuda.hpp

  tensorproducts/tensorproducts.hpp
  convolution/convolution.hpp
)

add_library(kernel_wrapper MODULE ${core_SOURCES} kernel_wrapper.cpp)
set_property(TARGET kernel_wrapper PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(kernel_wrapper PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/tensorproducts
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution
)

set_target_properties(kernel_wrapper PROPERTIES LINKER_LANGUAGE CXX)

# Manually build the module to avoid SOABI renaming. Original command:
# pybind11_add_module(kernel_wrapper kernel_wrapper.cpp)
# target_link_libraries(kernel_wrapper PRIVATE espmm)

target_link_libraries(kernel_wrapper PRIVATE 
    pybind11::pybind11
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc)
target_link_options(kernel_wrapper PRIVATE -Wl,-rpath='$ORIGIN')

if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(kernel_wrapper)
endif()

set_target_properties(kernel_wrapper PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden"
                                         PREFIX "")

install(TARGETS kernel_wrapper DESTINATION .)