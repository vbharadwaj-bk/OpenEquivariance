# Add library
add_library(espmm SHARED 
  espmm.hpp
  
  util/util.hpp
  util/gpu_util.hpp
  util/buffer.hpp
  util/buffer.cpp
  util/representation.hpp
  util/jit.hpp
  util/jit.cpp

  tensorproducts/tensorproducts.hpp
  tensorproducts/generic_tp.cpp
  tensorproducts/thread_tp.cu
  tensorproducts/gemm_tp.cu
  tensorproducts/shuffle_tp.cu
  tensorproducts/unroll_tp.cu

  convolution/convolution.hpp
  convolution/atomic_conv.cu
  convolution/generic_conv.cpp
)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

#find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS 
#  "${CMAKE_CURRENT_SOURCE_DIR}/../../nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/"
#)
#target_link_libraries(espmm mathdx::cublasdx)

target_include_directories(espmm PUBLIC 
  ${pybind11_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/tensorproducts
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution
  )

set(ESPMM_PUBLIC_HEADERS
    "espmm.hpp"
    "util/jit.hpp"
    "util/buffer.hpp"
    "util/util.hpp"
    "util/representation.hpp"
    "tensorproducts/tensorproducts.hpp"
    "convolution/convolution.hpp"
)

# Set target properties
set_target_properties(espmm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(espmm PROPERTIES PUBLIC_HEADER "${ESPMM_PUBLIC_HEADERS}")

# To-do: should make path to standard CUDA math libraries a parameter 
target_link_directories(espmm PUBLIC /opt/nvidia/hpc_sdk/Linux_x86_64/2024/math_libs/12.4/lib64/)
target_link_libraries(espmm CUDA::cudart -lcublasLt -lcuda CUDA::nvrtc)
install(TARGETS espmm PUBLIC_HEADER )
