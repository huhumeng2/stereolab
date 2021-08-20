#include "algorithm/stereo_match_bm.h"
#include "common/info_log.h"
#include "common/util.h"

#include <iostream>

namespace stereolab
{
namespace common
{

bool StereoMatchBM::configure(const nlohmann::json &config)
{
    config["stereo_bm"].get_to(param_);

    if (param_.cost_type > CostType::kSSD)
    {
        SL_PRINTE("Not implement cost type...\n");
        return false;
    }

    init_ = true;
    return true;
}

bool StereoMatchBM::compute(const StereoData &data, cv::Mat &disp)
{
    if (!init_)
        return false;

    if (data.left.empty() || data.right.empty() || (data.right.size() != data.left.size()))
    {
        SL_PRINTE("Image size not match required\n");
        return false;
    }

    const cv::Mat &im0 = data.left;
    const cv::Mat &im1 = data.right;

    disp.create(im0.size(), CV_16UC1);

    const int D = param_.max_disp - param_.min_disp;

    std::vector<float> cost_table(D, kInvalidCost);
    int block_size_square = param_.block_size * param_.block_size;

    for (int r = 0; r < im0.rows; ++r)
    {
        for (int c = 0; c < im0.cols; ++c)
        {
            for (int d = param_.min_disp; d < param_.max_disp; ++d)
            {
                if (c - d < 0)
                {
                    cost_table[d - param_.min_disp] = kInvalidCost;
                    continue;
                }

                float error = 0;

                for (int i = -param_.block_size / 2; i <= param_.block_size / 2; ++i)
                {
                    for (int j = -param_.block_size / 2; j <= param_.block_size / 2; ++j)
                    {
                        int y1 = std::min(std::max(0, r + i), im0.rows - 1);
                        int x1 = std::min(std::max(0, c + j), im0.cols - 1);
                        int x2 = std::min(std::max(0, c + j - d), im0.cols - 1);

                        if (param_.cost_type == CostType::kSAD)
                            error += l1_dist(im0.at<cv::Vec3b>(y1, x1), im1.at<cv::Vec3b>(y1, x2));
                        else if (param_.cost_type == CostType::kSSD)
                            error += std::sqrt(l2_dist(im0.at<cv::Vec3b>(y1, x1), im1.at<cv::Vec3b>(y1, x2)) / 3.0f);
                    }
                }

                if (param_.cost_type == CostType::kSAD)
                    cost_table[d - param_.min_disp] = error / (block_size_square * 255 * 3);
                else if (param_.cost_type == CostType::kSSD)
                    cost_table[d - param_.min_disp] = error / (block_size_square * 255);
            }

            float min_cost = cost_table[0];
            int best_d = 0;
            for (int d = 1; d < D; ++d)
            {
                if (cost_table[d] < min_cost)
                {
                    min_cost = cost_table[d];
                    best_d = d;
                }
            }

            float diff = 0.0f;
            if (best_d != 0 && best_d != D - 1)
            {
                float y1 = cost_table[best_d - 1];
                float y2 = cost_table[best_d];
                float y3 = cost_table[best_d + 1];

                diff = StereoMatch::sub_pixel_refine(y1, y2, y3);
            }

            disp.at<uint16_t>(r, c) = static_cast<uint16_t>(((best_d + param_.min_disp) << 4) + diff * 16.0f);
        }
    }

    return true;
}

}  // namespace common
}  // namespace stereolab