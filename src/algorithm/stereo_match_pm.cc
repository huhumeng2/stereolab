#include "algorithm/stereo_match_pm.h"
#include "common/info_log.h"
#include "common/util.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace stereolab
{
namespace algorithm
{

StereoMatchPM::StereoMatchPM() : StereoMatch() {}

StereoMatchPM::~StereoMatchPM() {}

bool StereoMatchPM::configure(const nlohmann::json &config)
{
    try
    {
        config["stereo_pm"].get_to(param_);
    }
    catch (const std::exception &e)
    {
        SL_PRINTE("%s\n", e.what());
        return false;
    }

    init_ = true;

    return true;
}

bool StereoMatchPM::compute(const common::StereoData &data, cv::Mat &disp)
{

    if (!init_)
    {
        SL_PRINTE("Init algorithm first!");
        return false;
    }

    if (!check_data_input(data))
    {
        SL_PRINTE("Image size not match required\n");
        return false;
    }

    image_[0] = data.left;
    image_[1] = data.right;

    disp.create(image_[0].size(), CV_32FC1);

    init_random_plane(image_[0].size(), disp_planes_[0], 1);
    init_random_plane(image_[1].size(), disp_planes_[1], -1);
    SL_PRINTI("Init random plane done\n");

    compute_pixel_weight(image_[0], weight_[0], param_.block_size);
    compute_pixel_weight(image_[1], weight_[1], param_.block_size);
    SL_PRINTI("Compute pixel weight done\n");

    cv::Mat gray0, gray1;
    cv::cvtColor(image_[0], gray0, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image_[1], gray1, cv::COLOR_BGR2GRAY);

    cv::Sobel(gray0, sobel_x_[0], CV_32FC1, 1, 0, 3);
    cv::Sobel(gray1, sobel_x_[1], CV_32FC1, 1, 0, 3);

    cv::Sobel(gray0, sobel_y_[0], CV_32FC1, 0, 1, 3);
    cv::Sobel(gray1, sobel_y_[1], CV_32FC1, 0, 1, 3);
    SL_PRINTI("Compute gradient done\n");

    convert_to_disp(disp_planes_[0], disp, 1);

    return true;
}

float StereoMatchPM::plane_match_cost(const DisparityPlane &plane, int side, int x, int y, int window_size)
{
    int half = window_size / 2;

    float cost = 0.0f;

    cv::Vec3b color0 = image_[side].at<cv::Vec3b>(y, x);

    float gx0 = sobel_x_[side].at<float>(y, x);
    float gy0 = sobel_y_[side].at<float>(y, x);

    for (int r = y - half; r <= y + half; ++r)
    {
        for (int c = x - half; c <= x + half; ++c)
        {
            if (r >= 0 && c >= 0 && r < image_[0].rows && c < image_[0].cols)
            {
                float disp = plane.disparity(r, c);

                // find matching point in other view
                float match = x - disp;

                if (match < 0 || match > image_[0].cols - 1)
                {
                    cost += kPenalty;
                    continue;
                }

                int xx = static_cast<int>(match);
                int xx1 = xx + 1;

                cv::Vec3b color_match0 = image_[1 - side].at<cv::Vec3b>(r, xx);
                cv::Vec3b color_match1 = image_[1 - side].at<cv::Vec3b>(r, xx1);

                float gx10 = sobel_x_[1 - side].at<float>(r, xx);
                float gx11 = sobel_x_[1 - side].at<float>(r, xx1);
                float gy10 = sobel_y_[1 - side].at<float>(r, xx);
                float gy11 = sobel_y_[1 - side].at<float>(r, xx1);

                float scale = match - static_cast<float>(xx);
                cv::Vec3f color_match = (1 - scale) * color_match0 + scale * color_match1;

                float gx1 = (1 - scale) * gx10 + scale * gx11;
                float gy1 = (1 - scale) * gy10 + scale * gy11;

                float w = weight_[side].at<float>(cv::Vec<int, 4>{ y, x, r - y + half, c - x + half });

                cost += w
                        * (std::min(param_.tau_c, common::l1_dist(color0, color_match)) * (1.0f - param_.alpha)
                           + std::min(param_.tau_g, std::abs(gx0 - gx1) + std::abs(gy0 - gy1)) * param_.alpha);
            }
        }
    }

    return cost;
}

void StereoMatchPM::evaluate_planes_cost(const cv::Mat &im0, const cv::Mat &im1,
                                         const cv::Mat_<DisparityPlane> disp_plane, cv::Mat &cost)
{
    cost.create(im0.size(), CV_32FC1);

#pragma omp parallel for
    for (int y = 0; y < im0.rows; ++y)
    {
        for (int x = 0; x < im0.cols; ++x)
        {
            cost.at<float>(y, x) = 1.0f;
        }
    }
}

void StereoMatchPM::init_random_plane(const cv::Size &size, cv::Mat_<DisparityPlane> &disp_planes, int sign)
{
    disp_planes.create(size);

    cv::RNG random_generator;

    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            float d = sign * random_generator.uniform(param_.min_disp, param_.max_disp);  // random disparity
            cv::Vec3d v3d = common::gen_random_unit_vec3d();

            auto &dp = disp_planes.at<DisparityPlane>(y, x);

            dp.nx = v3d(0);
            dp.ny = v3d(1);
            dp.nz = v3d(2);
            dp.z = d;
            dp.update_coeff(x, y, d);
        }
    }
}

void StereoMatchPM::convert_to_disp(const cv::Mat_<DisparityPlane> &disp_planes, cv::Mat &disp, int sign)
{
    if (disp.empty())
    {
        SL_PRINTW("Disparity image is empty\n");
        disp.create(disp_planes.size(), CV_32FC1);
    }

    for (int y = 0; y < disp_planes.rows; ++y)
    {
        for (int x = 0; x < disp_planes.cols; ++x)
        {
            disp.at<float>(y, x) = sign * disp_planes.at<DisparityPlane>(y, x).z;
        }
    }
}

float StereoMatchPM::color_weight(const cv::Vec3b &p, const cv::Vec3b &q) const
{
    return std::exp(-common::l1_dist(p, q) / param_.gamma);
}

void StereoMatchPM::compute_pixel_weight(const cv::Mat &image, cv::Mat &weight, int window_size)
{
    int weight_size[] = { image.rows, image.cols, window_size, window_size };
    weight.create(4, weight_size, CV_32F);

    int half = window_size / 2;

#pragma omp parallel for
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            for (int i = y - half; i <= y + half; ++i)
            {
                for (int j = x - half; j <= x + half; ++j)
                {
                    if (i >= 0 && i < image.rows && j >= 0 && j < image.cols)
                        weight.at<float>(cv::Vec<int, 4>{ y, x, i - y + half, j - x + half }) =
                            color_weight(image.at<cv::Vec3b>(y, x), image.at<cv::Vec3b>(i, j));
                }
            }
        }
    }
}

}  // namespace algorithm
}  // namespace stereolab