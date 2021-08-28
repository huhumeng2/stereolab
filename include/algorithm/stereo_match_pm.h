#pragma once

#include "algorithm/disparity_plane.h"
#include "algorithm/stereo_match.h"

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

    constexpr static float kPenalty = 120.0f;

public:
    StereoMatchPM();

    virtual ~StereoMatchPM();

    virtual bool configure(const nlohmann::json &config) override;

    virtual bool compute(const common::StereoData &data, cv::Mat &disp) override;

private:
    float color_weight(const cv::Vec3b &p, const cv::Vec3b &q) const;

    void compute_pixel_weight(const cv::Mat &image, cv::Mat &weight, int window_size);

    void init_random_plane(const cv::Size &size, cv::Mat_<DisparityPlane> &disp_planes, int sign);

    void convert_to_disp(const cv::Mat_<DisparityPlane> &disp_planes, cv::Mat &disp, int sign);

    float plane_match_cost(const DisparityPlane &plane, int side, int x, int y, int window_size);

    void evaluate_planes_cost(const cv::Mat &im0, const cv::Mat &im1,
                              const cv::Mat_<DisparityPlane> disp_plane,
                              cv::Mat &cost);

    cv::Mat weight_[2];
    cv::Mat image_[2];
    cv::Mat sobel_x_[2], sobel_y_[2];
    cv::Mat_<DisparityPlane> disp_planes_[2];

    PatchMatchParam param_;
};

}  // namespace algorithm
}  // namespace stereolab
