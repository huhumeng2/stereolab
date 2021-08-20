#include "algorithm/stereo_match_pm.h"
#include "common/info_log.h"
#include "common/util.h"

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

    const cv::Mat &im0 = data.left;
    const cv::Mat &im1 = data.right;

    disp.create(im0.size(), CV_32FC1);

    init_random_plane(im0.size());

    convert_to_disp(disp);

    return true;
}

void StereoMatchPM::init_random_plane(const cv::Size &size)
{
    disp_planes_.create(size);

    cv::RNG random_generator;

    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            float d = random_generator.uniform(param_.min_disp, param_.max_disp);  // random disparity
            cv::Vec3d v3d = common::gen_random_unit_vec3d();

            auto &dp = disp_planes_.at<DisparityPlane>(y, x);

            dp.nx = v3d(0);
            dp.ny = v3d(1);
            dp.nz = v3d(2);
            dp.z = d;
        }
    }
}

void StereoMatchPM::convert_to_disp(cv::Mat &disp)
{
    if (disp.empty())
    {
        SL_PRINTW("Disparity image is empty\n");
        disp.create(disp_planes_.size(), CV_32FC1);
    }

    for (int y = 0; y < disp_planes_.rows; ++y)
    {
        for (int x = 0; x < disp_planes_.cols; ++x)
        {
            disp.at<float>(y, x) = disp_planes_.at<DisparityPlane>(y, x).z;
        }
    }
}

}  // namespace algorithm
}  // namespace stereolab