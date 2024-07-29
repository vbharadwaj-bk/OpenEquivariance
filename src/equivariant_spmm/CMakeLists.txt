# Add library
#add_library(test_library SHARED 
#  test.cu
#)

add_executable(test_exec
  test.cu
)

# Set target properties
set_target_properties(test_exec PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(test_exec PROPERTIES LINKER_LANGUAGE CXX)
# Link against CUDA
target_link_libraries(test_exec ${CUDA_LIBRARIES})