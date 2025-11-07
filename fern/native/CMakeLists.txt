cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ilava_native LANGUAGES CXX CUDA)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# CUDA architecture support
# Volta (V100): 70
# Turing (T4, RTX 20xx): 75
# Ampere (A100, RTX 30xx): 80, 86
# Ada (RTX 40xx): 89
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# RVQ CUDA Library
add_library(rvq_cuda STATIC
    cuda/rvq_cuda.cu
)

set_target_properties(rvq_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(rvq_cuda
    ${CUDA_LIBRARIES}
    ${CUDA_cub_LIBRARY}
)

# Python bindings module
pybind11_add_module(rvq_cuda_ext
    python/bindings.cpp
)

target_link_libraries(rvq_cuda_ext PRIVATE
    rvq_cuda
    ${CUDA_LIBRARIES}
)

# Set output directory
set_target_properties(rvq_cuda_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
)

# Installation
install(TARGETS rvq_cuda_ext
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/ilava/native/python
)

# Testing (optional)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    # Add test executables here
endif()

message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Python: ${Python3_EXECUTABLE}")

