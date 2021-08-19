#pragma once

#include <opencv2/core/core.hpp>

namespace stereolab
{
namespace common
{

struct StereoData
{
    uint64_t index;
    cv::Mat left;
    cv::Mat right;
};

}  // namespace common
}  // namespace stereolab