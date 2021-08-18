#include <gtest/gtest.h>

#include "../test_config.h"
#include "common/util.h"
#include "common/info_log.h"

#include <filesystem>

#include <opencv2/highgui/highgui.hpp>

using namespace stereolab;
using namespace common;

TEST(EVALUATE_DISP_TEST, EVALUATE_DISP_SAME)
{
    auto im_path = std::filesystem::path(TEST_DATA_PATH) / "disp.png";
    cv::Mat disparity = cv::imread(im_path.string(), -1);
    ASSERT_FALSE(disparity.empty());

    SL_PRINTI("image resolution: %d x %d\n", disparity.cols, disparity.rows);
    cv::Mat disp_float;
    disparity.convertTo(disp_float, CV_32FC1);

    DispEvaluateResult eval_result;
    evaluate_disp(disp_float, disp_float, cv::Mat(), &eval_result, 256, 2.0f);

    eval_result.print_result();

    EXPECT_EQ(eval_result.bad_rate, 0.0);
    EXPECT_EQ(eval_result.invalid_rate, 0.0);
    EXPECT_EQ(eval_result.total_bad_rate, 0.0);
    EXPECT_EQ(eval_result.average_epe, 0.0);
    EXPECT_EQ(eval_result.number, disparity.rows * disparity.cols);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}