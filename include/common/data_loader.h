#pragma once

#include <string>
#include <vector>

#include "common/stereo_data.h"

namespace stereolab
{
namespace common
{
class DataLoader
{
public:
    DataLoader::DataLoader() : init_(false) {}

    virtual ~DataLoader() {}

    virtual bool configure(const std::string &data_folder) = 0;

    virtual bool get_stereo_data(StereoData *data) = 0;

    const std::string &data_folder() const { return data_folder_; }

    size_t data_size() const { return left_file_name_.size(); }

protected:
    bool init_;

    std::string data_folder_;
    std::vector<std::string> left_file_name_, right_file_name_;
};
}  // namespace common
}  // namespace stereolab