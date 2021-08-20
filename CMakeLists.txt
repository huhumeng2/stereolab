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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
else()
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
    )
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MT" "/MD" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()

    set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /std:c++17 /O2 /Zc:wchar_t-")
endif()

option(ENABLE_STEREO_LAB_DEBUG "Enable stereo lab debug code" On)
option(ENBALE_BUILD_TEST "Enable build test" On)

include_directories(${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/include/common/config.h.in 
    ${PROJECT_SOURCE_DIR}/include/common/config.h
)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(sl_common
    ${PROJECT_SOURCE_DIR}/src/common/util.cc
    ${PROJECT_SOURCE_DIR}/src/common/mb_data_loader.cc
)
target_include_directories(sl_common
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(sl_common
    PUBLIC
    ${OpenCV_LIBS}
)

add_library(sl_algorithm
    ${PROJECT_SOURCE_DIR}/src/algorithm/stereo_match_bm.cc
)
target_link_libraries(sl_algorithm
    PUBLIC
    sl_common
    nlohmann_json
)

if (${ENBALE_BUILD_TEST})
    enable_testing()
    add_subdirectory(test)
endif()
