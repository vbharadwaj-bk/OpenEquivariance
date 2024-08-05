# Add library
add_library(espmm SHARED 
  espmm.hpp
  espmm.cu
)

set(ESPMM_PUBLIC_HEADERS
    "espmm.hpp"
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
