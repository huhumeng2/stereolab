#pragma once

#include "common/device_error_handler.h"

namespace stereolab
{
namespace common
{

class DeviceMemManaged
{
public:
    void* operator new(size_t num_of_byte)
    {
        void* ptr;
        CheckCudaError(cudaMallocManaged(&ptr, num_of_byte));
        return ptr;
    }

    void operator delete(void* ptr)
    {
        CheckCudaError(cudaFree(ptr));
    }
};

}  // namespace common

}  // namespace stereolab
