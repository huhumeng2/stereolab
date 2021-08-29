#pragma once

#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace stereolab::algorithm
{

struct IMUState
{
    // An unique identifier for the IMU state.
    uint64_t id;

    // id for next IMU state
    static uint64_t next_id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to
    // the IMU (body) frame.
    Eigen::Vector4d orientation;

    // Position of the IMU (body) frame
    // in the world frame.
    Eigen::Vector3d position;

    // Velocity of the IMU (body) frame
    // in the world frame.
    Eigen::Vector3d velocity;

    // Bias for measured angular velocity
    // and acceleration.
    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    // Transformation between the IMU and the
    // left camera (cam0)
    Eigen::Matrix3d R_imu_cam0;
    Eigen::Vector3d t_cam0_imu;

    // These three variables should have the same physical
    // interpretation with `orientation`, `position`, and
    // `velocity`. There three variables are used to modify
    // the transition matrices to make the observability matrix
    // have proper null space.
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;
    Eigen::Vector3d velocity_null;

    // Process noise
    static double gyro_noise;
    static double acc_noise;
    static double gyro_bias_noise;
    static double acc_bias_noise;

    // Gravity vector in the world frame
    static Eigen::Vector3d gravity;

    // Transformation offset from the IMU frame to
    // the body frame. The transformation takes a
    // vector from the IMU frame to the body frame.
    // The z axis of the body frame should point upwards.
    // Normally, this transform should be identity.
    static Eigen::Isometry3d T_imu_body;

    IMUState()
        : id(0)
        , time(0.0)
        , orientation(Eigen::Vector4d(0, 0, 0, 1))
        , position(Eigen::Vector3d::Zero())
        , velocity(Eigen::Vector3d::Zero())
        , gyro_bias(Eigen::Vector3d::Zero())
        , acc_bias(Eigen::Vector3d::Zero())
        , orientation_null(Eigen::Vector4d(0, 0, 0, 1))
        , position_null(Eigen::Vector3d::Zero())
        , velocity_null(Eigen::Vector3d::Zero())
    {
    }

    IMUState(const uint64_t &new_id)
        : id(new_id)
        , time(0)
        , orientation(Eigen::Vector4d(0, 0, 0, 1))
        , position(Eigen::Vector3d::Zero())
        , velocity(Eigen::Vector3d::Zero())
        , gyro_bias(Eigen::Vector3d::Zero())
        , acc_bias(Eigen::Vector3d::Zero())
        , orientation_null(Eigen::Vector4d(0, 0, 0, 1))
        , position_null(Eigen::Vector3d::Zero())
        , velocity_null(Eigen::Vector3d::Zero())
    {
    }
};

struct CAMState
{
    // An unique identifier for the CAM state.
    uint64_t id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to the camera frame.
    Eigen::Vector4d orientation;

    // Position of the camera frame in the world frame.
    Eigen::Vector3d position;

    // These two variables should have the same physical
    // interpretation with `orientation` and `position`.
    // There two variables are used to modify the measurement
    // Jacobian matrices to make the observability matrix
    // have proper null space.
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;

    // Takes a vector from the cam0 frame to the cam1 frame.
    static Eigen::Isometry3d T_cam0_cam1;

    CAMState()
        : id(0)
        , time(0)
        , orientation(Eigen::Vector4d(0, 0, 0, 1))
        , position(Eigen::Vector3d::Zero())
        , orientation_null(Eigen::Vector4d(0, 0, 0, 1))
        , position_null(Eigen::Vector3d(0, 0, 0))
    {
    }

    CAMState(const uint64_t &new_id)
        : id(new_id)
        , time(0)
        , orientation(Eigen::Vector4d(0, 0, 0, 1))
        , position(Eigen::Vector3d::Zero())
        , orientation_null(Eigen::Vector4d(0, 0, 0, 1))
        , position_null(Eigen::Vector3d::Zero())
    {
    }
};

constexpr double kGravityAcceleration = 9.81;

// Static member variables in IMUState class.
inline uint64_t IMUState::next_id = 0;
inline double IMUState::gyro_noise = 0.001;
inline double IMUState::acc_noise = 0.01;
inline double IMUState::gyro_bias_noise = 0.001;
inline double IMUState::acc_bias_noise = 0.01;
inline Eigen::Vector3d IMUState::gravity = Eigen::Vector3d(0, 0, -kGravityAcceleration);
inline Eigen::Isometry3d IMUState::T_imu_body = Eigen::Isometry3d::Identity();

// Static member variables in CAMState class.
inline Eigen::Isometry3d CAMState::T_cam0_cam1 = Eigen::Isometry3d::Identity();

using CamStateServer = std::map<uint64_t, CAMState, std::less<uint64_t>>;
}  // namespace stereolab::algorithm
