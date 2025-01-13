find_package(pybind11)
find_package(Python COMPONENTS Interpreter Development)

add_library(espmm SHARED 
  espmm.hpp
  
  util/gpu_util.hpp
  util/buffer.hpp
  util/buffer.cpp
  util/jit.hpp
  util/jit.cpp
  util/device_prop.hpp
  util/device_prop.cpp
  util/mm_tester.hpp

  tensorproducts/tensorproducts.hpp
  tensorproducts/jit_tp.cpp

  convolution/convolution.hpp
  convolution/jit_conv.cpp
)

set_property(TARGET espmm PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(espmm PUBLIC 
  ${pybind11_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/tensorproducts
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution
  )

set_target_properties(espmm PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(espmm PUBLIC CUDA::cudart CUDA::nvrtc)

install(TARGETS espmm)