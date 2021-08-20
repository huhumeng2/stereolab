#pragma once

#include "common/data_loader.h"

namespace stereolab
{
namespace common
{
class MiddleburyDataLoader : public DataLoader
{
public:
    MiddleburyDataLoader() : DataLoader(), data_index_(0) {}

    ~MiddleburyDataLoader() {}

    virtual bool configure(const std::string &data_folder) override;

    virtual bool get_stereo_data(StereoData *data) override;

    bool get_gt_disp(int side, uint64_t index, cv::Mat *disp);

private:
    std::vector<std::string> disp1_file_name_, disp5_file_name_;
    uint32_t data_index_;
};
}  // namespace common
}  // namespace stereolab