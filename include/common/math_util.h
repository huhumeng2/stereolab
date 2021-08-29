#pragma once

#include <Eigen/Core>

namespace stereolab::common
{
/*
 *  @brief Create a skew-symmetric matrix from a 3-element vector.
 *  @note Performs the operation:
 *  w   ->  [  0 -w3  w2]
 *          [ w3   0 -w1]
 *          [-w2  w1   0]
 */
template <typename T>
inline Eigen::Matrix<T, 3, 3> skew_symmetric(const Eigen::Matrix<T, 3, 1> &w)
{
    Eigen::Matrix<T, 3, 3> w_hat;
    w_hat(0, 0) = T(0);
    w_hat(0, 1) = -w(2);
    w_hat(0, 2) = w(1);
    w_hat(1, 0) = w(2);
    w_hat(1, 1) = T(0);
    w_hat(1, 2) = -w(0);
    w_hat(2, 0) = -w(1);
    w_hat(2, 1) = w(0);
    w_hat(2, 2) = T(0);
    return w_hat;
}

/*
 * @brief Normalize the given quaternion to unit quaternion.
 */
template <typename T>
inline void quaternion_normalize(Eigen::Matrix<T, 4, 1> &q)
{
    double norm = q.norm();
    q = q / norm;
    return;
}

template <typename T>
inline auto quaternion_to_rotation(const Eigen::Matrix<T, 4, 1> &q)
{
    const Eigen::Matrix<T, 3, 1> &q_vec = q.head<3>();
    const T &q4 = q(3);
    Eigen::Matrix<T, 3, 3> R = (T(2) * q4 * q4 - T(1)) * Eigen::Matrix<T, 3, 3>::Identity()
                               - T(2) * q4 * skew_symmetric(q_vec) + T(2) * q_vec * q_vec.transpose();

    // TODO: Is it necessary to use the approximation equation
    //    (Equation (87)) when the rotation angle is small?
    return R;
}

/*
 * @brief Convert a rotation matrix to a quaternion.
 * @note Pay attention to the convention used. The function follows the
 *    conversion in "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for Quaternion Algebra", Equation (78).
 *
 *    The input quaternion should be in the form
 *      [q1, q2, q3, q4(scalar)]^T
 */
template <typename T>
inline auto rotation_to_quaternion(const Eigen::Matrix<T, 3, 3> &R)
{
    Eigen::Matrix<T, 4, 1> score;
    score(0) = R(0, 0);
    score(1) = R(1, 1);
    score(2) = R(2, 2);
    score(3) = R.trace();

    int max_row = 0, max_col = 0;
    score.maxCoeff(&max_row, &max_col);

    Eigen::Matrix<T, 4, 1> q = Eigen::Matrix<T, 4, 1>::Zero();
    if (max_row == 0)
    {
        q(0) = std::sqrt(T(1.0) + T(2.0) * R(0, 0) - R.trace()) / T(2.0);
        q(1) = (R(0, 1) + R(1, 0)) / (T(4.0) * q(0));
        q(2) = (R(0, 2) + R(2, 0)) / (T(4.0) * q(0));
        q(3) = (R(1, 2) - R(2, 1)) / (T(4.0) * q(0));
    }
    else if (max_row == 1)
    {
        q(1) = std::sqrt(T(1.0) + T(2.0) * R(1, 1) - R.trace()) / T(2.0);
        q(0) = (R(0, 1) + R(1, 0)) / (T(4.0) * q(1));
        q(2) = (R(1, 2) + R(2, 1)) / (T(4.0) * q(1));
        q(3) = (R(2, 0) - R(0, 2)) / (T(4.0) * q(1));
    }
    else if (max_row == 2)
    {
        q(2) = std::sqrt(T(1.0) + T(2.0) * R(2, 2) - R.trace()) / T(2.0);
        q(0) = (R(0, 2) + R(2, 0)) / (T(4.0) * q(2));
        q(1) = (R(1, 2) + R(2, 1)) / (T(4.0) * q(2));
        q(3) = (R(0, 1) - R(1, 0)) / (T(4.0) * q(2));
    }
    else
    {
        q(3) = std::sqrt(T(1.0) + R.trace()) / T(2.0);
        q(0) = (R(1, 2) - R(2, 1)) / (T(4.0) * q(3));
        q(1) = (R(2, 0) - R(0, 2)) / (T(4.0) * q(3));
        q(2) = (R(0, 1) - R(1, 0)) / (T(4.0) * q(3));
    }

    if (q(3) < 0)
        q = -q;
    quaternion_normalize(q);
    return q;
}

/*
 * @brief Perform q1 * q2
 */
template <typename T>
inline auto quaternion_multiplication(const Eigen::Matrix<T, 4, 1> &q1, const Eigen::Matrix<T, 4, 1> &q2)
{
    Eigen::Matrix<T, 4, 4> L;
    L(0, 0) = q1(3);
    L(0, 1) = q1(2);
    L(0, 2) = -q1(1);
    L(0, 3) = q1(0);
    L(1, 0) = -q1(2);
    L(1, 1) = q1(3);
    L(1, 2) = q1(0);
    L(1, 3) = q1(1);
    L(2, 0) = q1(1);
    L(2, 1) = -q1(0);
    L(2, 2) = q1(3);
    L(2, 3) = q1(2);
    L(3, 0) = -q1(0);
    L(3, 1) = -q1(1);
    L(3, 2) = -q1(2);
    L(3, 3) = q1(3);

    Eigen::Matrix<T, 4, 1> q = L * q2;
    quaternion_normalize(q);
    return q;
}

/*
 * @brief Convert the vector part of a quaternion to a
 *    full quaternion.
 * @note This function is useful to convert delta quaternion
 *    which is usually a 3x1 vector to a full quaternion.
 *    For more details, check Section 3.2 "Kalman Filter Update" in
 *    "Indirect Kalman Filter for 3D Attitude Estimation:
 *    A Tutorial for quaternion Algebra".
 */
template <typename T>
inline auto small_angle_quaternion(const Eigen::Matrix<T, 3, 1> &dtheta)
{

    Eigen::Matrix<T, 3, 1> dq = dtheta / T(2.0);
    Eigen::Matrix<T, 4, 1> q;

    T dq_square_norm = dq.squaredNorm();

    if (dq_square_norm <= T(1))
    {
        q.head<3>() = dq;
        q(3) = std::sqrt(1 - dq_square_norm);
    }
    else
    {
        q.head<3>() = dq;
        q(3) = T(1);
        q = q / std::sqrt(T(1) + dq_square_norm);
    }

    return q;
}
}  // namespace stereolab::common
