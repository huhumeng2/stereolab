#include <gtest/gtest.h>

#include "../test_config.h"
#include "algorithm/stereo_match_bm.h"
#include "algorithm/stereo_match_pm.h"
#include "common/info_log.h"
#include "common/util.h"

#include <filesystem>

#include <opencv2/highgui/highgui.hpp>

using namespace stereolab;
using namespace common;
using namespace algorithm;

class StereoMatchTestSuite : public testing::Test
{
public:
    virtual void SetUp() override
    {
        auto v1_path = std::filesystem::path(TEST_DATA_PATH) / "view1.png";
        auto v5_path = std::filesystem::path(TEST_DATA_PATH) / "view5.png";

        stereo_data.index = 0;
        stereo_data.left = cv::imread(v1_path.string(), cv::IMREAD_COLOR);
        stereo_data.right = cv::imread(v5_path.string(), cv::IMREAD_COLOR);

        auto gt_path = std::filesystem::path(TEST_DATA_PATH) / "disp.png";
        disp_gt = cv::imread(gt_path.string(), -1);
        disp_gt.convertTo(disp_gt_float, CV_32FC1, 1.0 / 3.0);
    }

    virtual void TearDown() override
    {
    }

    StereoData stereo_data;
    cv::Mat disp_gt, disp_gt_float;
};

TEST_F(StereoMatchTestSuite, STEREO_MATCH_BM)
{
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

    cv::Mat disp;
    ret = bm->compute(this->stereo_data, disp);
    ASSERT_TRUE(ret);

    cv::Mat disp_gt_float, disp_float;
    disp.convertTo(disp_float, CV_32FC1, 1.0 / 16.0);
    disp_gt.convertTo(disp_gt_float, CV_32FC1, 1.0 / 3.0);

    DispEvaluateResult result;
    evaluate_disp(disp_float, disp_gt_float, cv::Mat(), &result, 64.0, 2.0);

    SL_PRINTI("Stereo BM evaluate results: \n");
    result.print_result();

#if 0
    cv::Mat disp_gt_color = disp16_to_color(disp_gt, 211, 0);
    cv::imshow("disp_gt", disp_gt_color);

    cv::Mat imshow;
    cv::hconcat(stereo_data.left, stereo_data.right, imshow);
    cv::Mat disp_color = disp16_to_color(disp, 16 * 64, 0);
    cv::hconcat(imshow, disp_color, imshow);

    cv::imshow("left | right | disp", imshow);
    cv::waitKey(0);
#endif
}

TEST_F(StereoMatchTestSuite, STEREO_MATCH_PM)
{
    StereoMatchPM *pm = new StereoMatchPM;

    std::string config_string =
        "{"
        "\"stereo_pm\": {"
        "    \"alpha\": 0.8999999761581421,"
        "    \"block_size\": 11,"
        "    \"gamma\": 10.0,"
        "    \"iter_num\": 430,"
        "    \"max_disp\": 72,"
        "    \"min_disp\": 0,"
        "    \"tau_c\": 10.0,"
        "    \"tau_g\": 4.0"
        "}"
        "}";

    nlohmann::json config_json = nlohmann::json::parse(config_string);

    bool ret = pm->configure(config_json);
    ASSERT_TRUE(ret);

    cv::Mat disp;
    ret = pm->compute(this->stereo_data, disp);
    ASSERT_TRUE(ret);

    DispEvaluateResult result;
    evaluate_disp(disp, disp_gt_float, cv::Mat(), &result, 72.0, 2.0);

    SL_PRINTI("Stereo PM evaluate results: \n");
    result.print_result();

#if 1

    cv::Mat imshow;
    cv::hconcat(stereo_data.left, stereo_data.right, imshow);
    cv::Mat disp_color = disp16_to_color(disp, 72.0, 0);
    cv::hconcat(imshow, disp_color, imshow);

    cv::imshow("left | right | disp", imshow);
    cv::waitKey(0);

#endif

    delete pm;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}