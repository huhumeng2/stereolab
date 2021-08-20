#pragma once

#include "algorithm/stereo_match.h"
#include "algorithm/disparity_plane.h"

namespace stereolab
{
namespace algorithm
{

struct PatchMatchParam
{

    PatchMatchParam()
        : min_disp(0),
          max_disp(64),
          block_size(3),
          iter_num(3),
          tau_c(10.0f),
          tau_g(2.0f),
          alpha(0.9f),
          gamma(10.0f)
    {
    }

    int32_t min_disp;
    int32_t max_disp;

    int32_t block_size;
    int32_t iter_num;

    float tau_c;
    float tau_g;

    float alpha;
    float gamma;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PatchMatchParam,
                                   min_disp,
                                   max_disp,
                                   block_size,
                                   iter_num,
                                   tau_c,
                                   tau_g,
                                   alpha,
                                   gamma)

class StereoMatchPM : public StereoMatch
{
public:
    StereoMatchPM();

    virtual ~StereoMatchPM();

    virtual bool configure(const nlohmann::json &config) override;

    virtual bool compute(const common::StereoData &data, cv::Mat &disp) override;

private:
    void init_random_plane(const cv::Size &size);

    void convert_to_disp(cv::Mat &disp);

    cv::Mat_<DisparityPlane> disp_planes_;

    PatchMatchParam param_;
};

}  // namespace algorithm
}  // namespace stereolab

