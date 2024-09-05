# Add library
add_library(espmm SHARED 
  espmm.hpp
  
  util/util.hpp
  util/gpu_util.hpp
  util/buffer.hpp
  util/buffer.cpp
  util/representation.hpp

  tensorproducts/tensorproducts.hpp
  tensorproducts/generic_tp.cpp
  tensorproducts/thread_tp.cu
  tensorproducts/gemm_tp.cu

  convolution/convolution.hpp
  convolution/equivariant_convolution.cu
)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)

#find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS 
#  "${CMAKE_CURRENT_SOURCE_DIR}/../../nvidia-mathdx-24.08.0/nvidia/mathdx/24.08/"
#)
#target_link_libraries(espmm mathdx::cublasdx)

set(ESPMM_PUBLIC_HEADERS
    "espmm.hpp"
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
target_link_libraries(espmm CUDA::cudart -lcublasLt)
install(TARGETS espmm PUBLIC_HEADER)
