cmake_minimum_required(VERSION 3.26)
project(pmpp LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_compile_options(
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

add_subdirectory(utils)
add_subdirectory(chapters/ch02)
add_subdirectory(chapters/ch03)
