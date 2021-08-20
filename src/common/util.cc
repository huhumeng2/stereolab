#include "common/util.h"
#include "common/info_log.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace stereolab
{
namespace common
{

void DispEvaluateResult::print_result()
{
    SL_PRINTI("number  bad_rate invalid_rate all_bad_rate avg_error\n")
    SL_PRINTI("%8d  %2.2f  %6.2f  %6.2f  %6.2f\n", number,
              bad_rate, invalid_rate, total_bad_rate, average_epe);
}

void evaluate_disp(const cv::Mat &disp, const cv::Mat &disp_gt, const cv::Mat &mask,
                   DispEvaluateResult *result, float max_valid, float bad_ths)
{

    if (result == nullptr)
    {
        SL_PRINTE("result pointer is null");
        return;
    }

    if (disp.size() != disp_gt.size())
    {
        SL_PRINTE("disparity size don't match, disp[%d x %d] vs disp_gt[%d x %d]",
                  disp.cols, disp.rows, disp_gt.cols, disp_gt.rows);
        return;
    }

    if (disp.type() != CV_32FC1 || disp_gt.type() != CV_32FC1)
    {
        SL_PRINTE("evalute only support image type CV_32FC1, disp[%d] vs disp_gt[%d]",
                  disp.type(), disp_gt.type());
        return;
    }

    bool use_mask = (!mask.empty());

    if (use_mask && disp.size() != mask.size() && mask.type() != CV_8UC1)
    {
        SL_PRINTE("use mask but input is invalid mask[%d x %d] type[%d]",
                  mask.cols, mask.rows, mask.type());
        return;
    }

    int width = disp.cols, height = disp.rows;

    int n = 0;
    int bad = 0;
    int invalid = 0;

    float sum_err = 0.0f;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float gt = disp_gt.at<float>(y, x);

            if (gt >= max_valid || gt == 0)
                continue;

            float d = disp.at<float>(y, x);

            bool valid = (d < max_valid);

            if (!use_mask || mask.at<uint8_t>(y, x) == 255)
            {
                n++;

                if (valid)
                {
                    float error = std::abs(d - gt);
                    sum_err += error;

                    if (error > bad_ths)
                    {
                        bad++;
                    }
                }
                else
                    invalid++;
            }
        }
    }

    result->bad_rate = 100.0f * bad / (n + 1);
    result->invalid_rate = 100.0f * invalid / (n + 1);
    result->total_bad_rate = 100.0f * (bad + invalid) / (n + 1);
    result->average_epe = sum_err / (n - invalid + 1);
    result->number = n;
}

float l1_dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
    cv::Vec3f af = a;
    cv::Vec3f bf = b;

    cv::Vec3f diff = af - bf;

    return std::abs(diff(0)) + std::abs(diff(1)) + std::abs(diff(2));
}

float l2_dist(const cv::Vec3b &a, const cv::Vec3b &b)
{
    cv::Vec3f af = a;
    cv::Vec3f bf = b;

    cv::Vec3f diff = af - bf;
    
    return diff.dot(diff);;
}

cv::Mat disp16_to_color(const cv::Mat disp_16, uint16_t max, uint16_t min)
{
    cv::Mat disp8u;
    cv::Mat disp = cv::min(cv::max(disp_16, min), max);
    disp.convertTo(disp8u, CV_8UC1, 255.0 / (max - min), -255.0 * min / (max - min));

    cv::Mat disp_color;
    cv::applyColorMap(disp8u, disp_color, cv::COLORMAP_JET);

    return disp_color;
}
}  // namespace common
}  // namespace stereolab