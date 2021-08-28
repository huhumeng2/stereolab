#pragma once

#include "common/stereo_data.h"

#include <nlohmann/json.hpp>

namespace stereolab
{
namespace algorithm
{
enum class CostType : uint32_t
{
    kSAD = 0,
    kSSD,
    kNCC,
    kZNCC,
    kCENSUS
};

NLOHMANN_JSON_SERIALIZE_ENUM(CostType, { { CostType::kSAD, "SAD" },
                                         { CostType::kSSD, "SSD" },
                                         { CostType::kNCC, "NCC" },
                                         { CostType::kZNCC, "ZNCC" },
                                         { CostType::kCENSUS, "CENSUS" } })

class StereoMatch
{
public:
    StereoMatch::StereoMatch() : init_(false) {}

    virtual bool configure(const nlohmann::json &config) = 0;

    virtual bool compute(const common::StereoData &data, cv::Mat &disp) = 0;

protected:

    bool check_data_input(const common::StereoData &data) const
    {
        if (data.left.empty() || data.right.empty() || (data.right.size() != data.left.size()))
        {
            return false;
        }

        return true;
    }

    template <typename T>
    static float sub_pixel_refine(T cost1, T cost2, T cost3)
    {
        float a = cost3 + cost1 - 2.0f * cost2;
        return (cost1 - cost3) / (2.0f * a);
    }

    bool init_;
};

}  // namespace algorithm
}  // namespace stereolab