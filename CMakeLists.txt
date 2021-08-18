cmake_minimum_required(VERSION 3.10)
project(
    StereoLab 
    VERSION 0.0.1 
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=compute_75;-code=compute_75")

message(STATUS "Operating System: " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiler ID: " ${CMAKE_CXX_COMPILER_ID})

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
else()
    set(CMAKE_CXX_FLAGS "/MT /W4")
endif()

option(ENABLE_STEREO_LAB_DEBUG "Enable stereo lab debug code" On)
option(ENBALE_BUILD_TEST "Enable build test" On)

include_directories(${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/include/common/config.h.in 
    ${PROJECT_SOURCE_DIR}/include/common/config.h
)

find_package(OpenCV REQUIRED)

if (${ENBALE_BUILD_TEST})
    enable_testing()
    add_subdirectory(test)
endif()
