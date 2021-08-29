#pragma once

#include <cstdint>
#include <map>

#include <Eigen/Core>

#include "algorithm/state.h"

namespace stereolab::algorithm
{
struct Feature
{

    /*
     * @brief OptimizationConfig Configuration parameters
     *    for 3d feature position optimization.
     */
    struct OptimizationConfig
    {
        double translation_threshold;
        double huber_epsilon;
        double estimation_precision;
        double initial_damping;
        int outer_loop_max_iteration;
        int inner_loop_max_iteration;

        OptimizationConfig()
            : translation_threshold(0.2)
            , huber_epsilon(0.01)
            , estimation_precision(5e-7)
            , initial_damping(1e-3)
            , outer_loop_max_iteration(10)
            , inner_loop_max_iteration(10)
        {
            return;
        }
    };

    // Constructors for the struct.
    Feature() : id(0), position(Eigen::Vector3d::Zero()), is_initialized(false) {}

    Feature(const uint64_t &new_id) : id(new_id), position(Eigen::Vector3d::Zero()), is_initialized(false) {}

    /*
     * @brief cost Compute the cost of the camera observations
     * @param T_c0_c1 A rigid body transformation takes
     *    a vector in c0 frame to ci frame.
     * @param x The current estimation.
     * @param z The ith measurement of the feature j in ci frame.
     * @return e The cost of this observation.
     */
    void cost(const Eigen::Isometry3d &T_c0_ci, const Eigen::Vector3d &x, const Eigen::Vector2d &z, double &e) const;

    /*
     * @brief jacobian Compute the Jacobian of the camera observation
     * @param T_c0_c1 A rigid body transformation takes
     *    a vector in c0 frame to ci frame.
     * @param x The current estimation.
     * @param z The actual measurement of the feature in ci frame.
     * @return J The computed Jacobian.
     * @return r The computed residual.
     * @return w Weight induced by huber kernel.
     */
    void jacobian(const Eigen::Isometry3d &T_c0_ci, const Eigen::Vector3d &x, const Eigen::Vector2d &z,
                  Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r, double &w) const;

    /*
     * @brief InitializePosition Intialize the feature position
     *    based on all current available measurements.
     * @param cam_states: A map containing the camera poses with its
     *    ID as the associated key value.
     * @return The computed 3d position is used to set the position
     *    member variable. Note the resulted position is in world
     *    frame.
     * @return True if the estimated 3d position of the feature
     *    is valid.
     */
    bool initialize_position(const CamStateServer &cam_states);

    /*
     * @brief generate_initial_guess Compute the initial guess of
     *    the feature's 3d position using only two views.
     * @param T_c1_c2: A rigid body transformation taking
     *    a vector from c2 frame to c1 frame.
     * @param z1: feature observation in c1 frame.
     * @param z2: feature observation in c2 frame.
     * @return p: Computed feature position in c1 frame.
     */
    void generate_initial_guess(const Eigen::Isometry3d &T_c1_c2, const Eigen::Vector2d &z1, const Eigen::Vector2d &z2,
                                Eigen::Vector3d &p) const;

    /*
     * @brief check_motion Check the input camera poses to ensure
     *    there is enough translation to triangulate the feature
     *    positon.
     * @param cam_states : input camera poses.
     * @return True if the translation between the input camera
     *    poses is sufficient.
     */
    bool check_motion(const CamStateServer &cam_states) const;

    // An unique identifier for the feature.
    // In case of long time running, the variable
    // type of id is set to FeatureIDType in order
    // to avoid duplication.
    uint64_t id;

    // id for next feature
    static uint64_t next_id;

    // Store the observations of the features in the
    // state_id(key)-image_coordinates(value) manner.
    std::map<uint64_t, Eigen::Vector4d, std::less<uint64_t>> observations;

    // 3d postion of the feature in the world frame.
    Eigen::Vector3d position;

    // A indicator to show if the 3d postion of the feature
    // has been initialized or not.
    bool is_initialized;

    // Noise for a normalized feature measurement.
    static double observation_noise;

    // Optimization configuration for solving the 3d position.
    static OptimizationConfig optimization_config;
};

// Static member variables in Feature class.
inline uint64_t Feature::next_id = 0;
inline double Feature::observation_noise = 0.01;
inline Feature::OptimizationConfig Feature::optimization_config;
}  // namespace stereolab::algorithm
