#include <gtest/gtest.h>

#include "../test_config.h"
#include "common/info_log.h"
#include "common/math_util.h"
#include "common/util.h"

#include <filesystem>

#include <Eigen/Dense>
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
}

TEST(UNIT_RANDOM_VECTOR_TEST, RAMDOM_VECTOR)
{
    for (int i = 0; i < 10000; ++i)
    {
        cv::Vec3d v = gen_random_unit_vec3d();
        double norm2 = v.dot(v);
        EXPECT_NEAR(norm2, 1.0, 1e-8);
    }
}

TEST(MATH_UTILS_TEST, SKEW_SYMMETRIC)
{
    Eigen::Vector3d w(1.0, 2.0, 3.0);
    Eigen::Matrix3d w_hat = skew_symmetric(w);
    Eigen::Vector3d zero_vector = w_hat * w;

    Eigen::FullPivLU<Eigen::Matrix3d> lu_helper(w_hat);

    EXPECT_EQ(lu_helper.rank(), 2);
    EXPECT_DOUBLE_EQ(zero_vector.norm(), 0.0);
    return;
}

TEST(MATH_UTILS_TEST, QUATERNION_NORMALIZE)
{
    Eigen::Vector4d q = Eigen::Vector4d::Random();
    quaternion_normalize(q);

    EXPECT_DOUBLE_EQ(q.norm(), 1.0);
    return;
}

TEST(MATH_UTILS_TEST, QUATERNION_TO_ROTATION)
{
    Eigen::Vector4d q(0.0, 0.0, 0.0, 1.0);
    Eigen::Matrix3d zero_matrix = quaternion_to_rotation(q) - Eigen::Matrix3d::Identity();

    Eigen::FullPivLU<Eigen::Matrix3d> lu_helper(zero_matrix);
    EXPECT_EQ(lu_helper.rank(), 0);
    return;
}

TEST(MATH_UTILS_TEST, ROTATION_TO_QUATERNION)
{
    Eigen::Vector4d q1(0.0, 0.0, 0.0, 1.0);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Vector4d q2 = rotation_to_quaternion(I);
    Eigen::Vector4d zero_vector = q1 - q2;

    EXPECT_DOUBLE_EQ(zero_vector.norm(), 0.0);
    return;
}

TEST(MATH_UTILS_TEST, QUATERNION_MULTIPLICATION)
{
    Eigen::Vector4d q1(2.0, 2.0, 1.0, 1.0);
    Eigen::Vector4d q2(1.0, 2.0, 3.0, 1.0);
    quaternion_normalize(q1);
    quaternion_normalize(q2);
    Eigen::Vector4d q_prod = quaternion_multiplication(q1, q2);

    Eigen::Matrix3d R1 = quaternion_to_rotation(q1);
    Eigen::Matrix3d R2 = quaternion_to_rotation(q2);
    Eigen::Matrix3d R_prod = R1 * R2;
    Eigen::Matrix3d R_prod_cp = quaternion_to_rotation(q_prod);

    Eigen::Matrix3d zero_matrix = R_prod - R_prod_cp;

    EXPECT_NEAR(zero_matrix.sum(), 0.0, 1e-10);
    return;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}