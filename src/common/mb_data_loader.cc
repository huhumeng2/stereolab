#include "common/mb_data_loader.h"
#include "common/info_log.h"

#include <filesystem>
#include <iostream>

#include <opencv2/highgui/highgui.hpp>

namespace stereolab
{
namespace common
{
bool MiddleburyDataLoader::configure(const std::string &data_folder)
{

    data_folder_ = data_folder;

    for (auto dir : std::filesystem::directory_iterator(data_folder))
    {
        auto p1 = dir.path() / "view1.png";
        auto p2 = dir.path() / "view5.png";
        auto p3 = dir.path() / "disp1.png";
        auto p4 = dir.path() / "disp5.png";

        auto flag1 = std::filesystem::exists(p1);
        auto flag2 = std::filesystem::exists(p2);
        auto flag3 = std::filesystem::exists(p3);
        auto flag4 = std::filesystem::exists(p4);

        if (flag1 && flag2 && flag3 && flag4)
        {
            left_file_name_.emplace_back(p1.string());
            right_file_name_.emplace_back(p2.string());

            disp1_file_name_.emplace_back(p3.string());
            disp5_file_name_.emplace_back(p4.string());
        }
    }

    init_ = true;

    return true;
}

bool MiddleburyDataLoader::get_stereo_data(StereoData *data)
{
    if (!init_)
    {
        return false;
    }

    if (data_index_ >= data_size())
    {
        return false;
    }

    auto &l_name = left_file_name_.at(data_index_);
    auto &r_name = right_file_name_.at(data_index_);

    data->left = cv::imread(l_name, cv::IMREAD_COLOR);
    data->right = cv::imread(r_name, cv::IMREAD_COLOR);
    data->index = static_cast<uint64_t>(data_index_);

    data_index_++;

    if (data->left.empty() || data->right.empty())
    {
        SL_PRINTE("Unexpected empty image please check");
        return false;
    }

    if (data->left.size() != data->right.size())
    {

        SL_PRINTE("Unmatched image size");
        return false;
    }

    return true;
}

bool MiddleburyDataLoader::get_gt_disp(int side, uint64_t index, cv::Mat *disp)
{

    if (index >= static_cast<uint64_t>(data_size()))
    {
        return false;
    }

    std::string disp_name;

    if (side == 0)
    {
        disp_name = disp1_file_name_.at(index);
    }
    else if (side == 1)
    {
        disp_name = disp5_file_name_.at(index);
    }
    else
    {
        SL_PRINTE("Invalid side, can only be 0 or 1");
        return false;
    }

    *disp = cv::imread(disp_name, -1);

    return !disp->empty();
}

}  // namespace common
}  // namespace stereolab