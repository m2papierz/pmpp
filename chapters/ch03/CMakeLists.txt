# Chapter 3 targets
add_executable(grayscale grayscale.cu)
add_executable(blur blur.cu)
add_executable(matmul matmul.cu)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Link to common utilities and image I/O
target_link_libraries(grayscale PRIVATE utils ${OpenCV_LIBS})
target_link_libraries(blur PRIVATE utils ${OpenCV_LIBS})
target_link_libraries(matmul PRIVATE utils)

target_include_directories(grayscale PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(blur PRIVATE ${OpenCV_INCLUDE_DIRS})

# Per-target CUDA settings (inherit globals from root)
set_target_properties(grayscale PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
