#include <gtest/gtest.h>

#include "../test_config.h"

#include "common/info_log.h"
#include "common/mb_data_loader.h"

using namespace stereolab;
using namespace common;

TEST(DATALOADER_TEST, EVALUATE_MIDDLEBURY)
{
    MiddleburyDataLoader *dataloader = new MiddleburyDataLoader;
    dataloader->configure("D:\\Dataset\\Middlebury\\Middlebury2");

    SL_PRINTI("Loaded %zd stereo images from %s\n", dataloader->data_size(), dataloader->data_folder().c_str());

    StereoData data;
    cv::Mat disp;
    for (size_t i = 0; i < dataloader->data_size(); ++i)
    {
        bool ret = dataloader->get_stereo_data(&data);
        EXPECT_TRUE(ret);

        ret = dataloader->get_gt_disp(0, i, &disp);
        EXPECT_TRUE(ret);

        ret = dataloader->get_gt_disp(1, i, &disp);
        EXPECT_TRUE(ret);
    }

    bool ret = dataloader->get_stereo_data(&data);
    EXPECT_FALSE(ret);

    ret = dataloader->get_gt_disp(0, dataloader->data_size(), &disp);
    EXPECT_FALSE(ret);

    ret = dataloader->get_gt_disp(1, dataloader->data_size(), &disp);
    EXPECT_FALSE(ret);

    delete dataloader;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}