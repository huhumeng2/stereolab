cmake_minimum_required(VERSION 3.10)
project(
    PatchMatchCuda 
    VERSION 0.0.1 
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_75;-code=compute_75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

option(ENABLE_STEREO_LAB_DEBUG "Enable stereo lab debug code" On)

include_directories(${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/include/common/config.h.in 
    ${PROJECT_SOURCE_DIR}/include/common/config.h
)

find_package(OpenCV REQUIRED)
