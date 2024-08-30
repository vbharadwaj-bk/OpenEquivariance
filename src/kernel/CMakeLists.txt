# Add library
add_library(espmm SHARED 
  espmm.hpp

  util/util.hpp
  util/gpu_util.hpp
  util/buffer.hpp
  util/buffer.cpp

  tensorproducts/tensorproducts.hpp
  tensorproducts/generic_tp.cpp
  tensorproducts/thread_tp.cu

  convolution/convolution.hpp
  convolution/equivariant_convolution.cu
)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)

set(ESPMM_PUBLIC_HEADERS
    "espmm.hpp"
    "util/buffer.hpp"
    "util/util.hpp"
    "util/util.hpp"
    "tensorproducts/tensorproducts.hpp"
    "convolution/convolution.hpp"
)

# Set target properties
set_target_properties(espmm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(espmm PROPERTIES PUBLIC_HEADER "${ESPMM_PUBLIC_HEADERS}")

# Link against CUDA
target_link_libraries(espmm ${CUDA_LIBRARIES})
install(TARGETS espmm PUBLIC_HEADER)
