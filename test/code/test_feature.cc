#include <gtest/gtest.h>

#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "algorithm/feature.h"
#include "algorithm/state.h"
#include "common/math_util.h"

using namespace stereolab;
using namespace algorithm;
using namespace common;

TEST(FEATURE_INIT_TEST, SPHERE_DISTRIBUTION)
{
    // Set the real feature at the origin of the world frame.
    Eigen::Vector3d feature(0.5, 0.0, 0.0);

    // Add 6 camera poses, all of which are able to see the
    // feature at the origin. For simplicity, the six camera
    // view are located at the six intersections between a
    // unit sphere and the coordinate system. And the z axes
    // of the camera frames are facing the origin.
    std::vector<Eigen::Isometry3d> cam_poses(6);

    // Positive x axis.
    cam_poses[0].linear() << 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    cam_poses[0].translation() << 1.0, 0.0, 0.0;
    // Positive y axis.
    cam_poses[1].linear() << -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0;
    cam_poses[1].translation() << 0.0, 1.0, 0.0;
    // Negative x axis.
    cam_poses[2].linear() << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    cam_poses[2].translation() << -1.0, 0.0, 0.0;
    // Negative y axis.
    cam_poses[3].linear() << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0;
    cam_poses[3].translation() << 0.0, -1.0, 0.0;
    // Positive z axis.
    cam_poses[4].linear() << 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
    cam_poses[4].translation() << 0.0, 0.0, 1.0;
    // Negative z axis.
    cam_poses[5].linear() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    cam_poses[5].translation() << 0.0, 0.0, -1.0;

    // Set the camera states
    CamStateServer cam_states;
    for (int i = 0; i < 6; ++i)
    {
        CAMState new_cam_state;
        new_cam_state.id = static_cast<uint64_t>(i);
        new_cam_state.time = static_cast<double>(i);
        new_cam_state.orientation = rotation_to_quaternion(Eigen::Matrix3d(cam_poses[i].linear().transpose()));
        new_cam_state.position = cam_poses[i].translation();
        cam_states[new_cam_state.id] = new_cam_state;
    }

    std::vector<Eigen::Vector4d> measurements(6);
    std::mt19937 engine;
    std::normal_distribution<double> gaussian(0.0, 0.01);

    for (int i = 0; i < 6; ++i)
    {
        Eigen::Isometry3d cam_pose_inv = cam_poses[i].inverse();
        Eigen::Vector3d p = cam_pose_inv.linear() * feature + cam_pose_inv.translation();

        double u = p(0) / p(2) + gaussian(engine);
        double v = p(1) / p(2) + gaussian(engine);

        measurements[i] = Eigen::Vector4d(u, v, u, v);
    }

    for (int i = 0; i < 6; ++i)
    {
        std::cout << "pose " << i << ":" << std::endl;
        std::cout << "orientation: " << std::endl;
        std::cout << cam_poses[i].linear() << std::endl;
        std::cout << "translation: " << std::endl;
        std::cout << cam_poses[i].translation().transpose() << std::endl;
        std::cout << "measurement: " << std::endl;
        std::cout << measurements[i].transpose() << std::endl;
        std::cout << std::endl;
    }

    // Initialize a feature object.
    Feature feature_object;
    for (int i = 0; i < 6; ++i)
        feature_object.observations[i] = measurements[i];

    // Compute the 3d position of the feature.
    feature_object.initialize_position(cam_states);

    // Check the difference between the computed 3d
  // feature position and the groud truth.
  std::cout << "ground truth position: " << feature.transpose() << std::endl;
  std::cout << "estimated position: " << feature_object.position.transpose() << std::endl;
  Eigen::Vector3d error = feature_object.position - feature;
  EXPECT_NEAR(error.norm(), 0, 0.05);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
