#pragma once

#include <opencv2/core/core.hpp>

namespace stereolab
{
namespace common
{
struct DispEvaluateResult
{
    float bad_rate;
    float invalid_rate;
    float total_bad_rate;
    float average_epe;

    int number;

    void print_result();
};

void evaluate_disp(const cv::Mat &disp, const cv::Mat &disp_gt, const cv::Mat &mask,
                   DispEvaluateResult *result, float max_valid, float bad_ths);

float l1_dist(const cv::Vec3b &a, const cv::Vec3b &b);
float l2_dist(const cv::Vec3b &a, const cv::Vec3b &b);

cv::Mat disp16_to_color(const cv::Mat disp_16, uint16_t max, uint16_t min);

}  // namespace common
}  // namespace stereolab