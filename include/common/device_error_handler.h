#pragma once

#include "common/config.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

namespace stereolab
{
namespace common
{
namespace detail
{
inline cudaError_t check_cuda_error(cudaError_t error_code, const char* const file, const int line)
{
#if STEREO_LAB_DEBUG_FLAG
    if(error_code != cudaSuccess)
    {
        fprintf(stderr, "[file : %s line : %d] CUDA Runtime Error : %s\n", file, line, cudaGetErrorString(error_code));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
#endif
    return error_code;
}

inline void check_last_cuda_error(const char* const file, const int line)
{
#if STEREO_LAB_DEBUG_FLAG
    cudaError_t error_code = cudaGetLastError();
    if(error_code != cudaSuccess)
    {
        fprintf(stderr, "[file : %s, line : %d] CUDA Error : %s\n", file, line, cudaGetErrorString(error_code));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    error_code = cudaDeviceSynchronize();
    if(error_code != cudaSuccess)
    {
        fprintf(stderr, "[file %s, line : %d] CUDA Sync Error : %s\n", file, line, cudaGetErrorString(error_code));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
#endif
}
}  // namespace detail

#define CheckCudaError(val) detail::check_cuda_error((val), __FILE__, __LINE__)
#define checkForLastCudaError() detail::check_last_cuda_error(__FILE__, __LINE__)

}  // namespace common
}  // namespace stereolab
