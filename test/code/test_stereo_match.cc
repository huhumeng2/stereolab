#include <gtest/gtest.h>

#include "../test_config.h"
#include "algorithm/stereo_match_bm.h"
#include "common/info_log.h"
#include "common/util.h"

#include <filesystem>

#include <opencv2/highgui/highgui.hpp>

using namespace stereolab;
using namespace common;
using namespace algorithm;

TEST(STEREO_MATCH_TEST, STEREO_MATCH_BM)
{
    auto v1_path = std::filesystem::path(TEST_DATA_PATH) / "view1.png";
    auto v5_path = std::filesystem::path(TEST_DATA_PATH) / "view5.png";

    StereoMatchBM *bm = new StereoMatchBM;

    std::string json_config =
        "{"
        "\"stereo_bm\": {"
        "\"block_size\": 3,"
        "\"cost_type\": \"SAD\","
        "\"max_disp\": 64,"
        "\"min_disp\": 0"
        "}"
        "}";

    nlohmann::json bm_config = nlohmann::json::parse(json_config);

    bool ret = bm->configure(bm_config);
    ASSERT_TRUE(ret);

    StereoData data;
    data.index = 0;
    data.left = cv::imread(v1_path.string(), cv::IMREAD_COLOR);
    data.right = cv::imread(v5_path.string(), cv::IMREAD_COLOR);

    cv::Mat disp;
    ret = bm->compute(data, disp);
    ASSERT_TRUE(ret);

    auto gt_path = std::filesystem::path(TEST_DATA_PATH) / "disp.png";
    cv::Mat disp_gt = cv::imread(gt_path.string(), -1);

    double min_value, max_value;
    cv::minMaxIdx(disp_gt, &min_value, &max_value);
    SL_PRINTI("min disp = %lf, max disp = %lf\n", min_value, max_value);

    cv::Mat disp_gt_float, disp_float;
    disp.convertTo(disp_float, CV_32FC1, 1.0 / 16.0);
    disp_gt.convertTo(disp_gt_float, CV_32FC1, 1.0 / 3.0);

    DispEvaluateResult result;
    evaluate_disp(disp_float, disp_gt_float, cv::Mat(), &result, 64.0, 2.0);
    result.print_result();
#if 0
    cv::Mat disp_gt_color = disp16_to_color(disp_gt, static_cast<uint16_t>(max_value), 0);
    cv::imshow("disp_gt", disp_gt_color);

    cv::Mat imshow;
    cv::hconcat(data.left, data.right, imshow);
    cv::Mat disp_color = disp16_to_color(disp, 16 * 64, 0);
    cv::hconcat(imshow, disp_color, imshow);

    cv::imshow("left | right | disp", imshow);
    cv::waitKey(0);
#endif
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}