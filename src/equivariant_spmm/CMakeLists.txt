# Add library
add_library(test_library SHARED 
  test.cu
)

# Set target properties
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})