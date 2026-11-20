"""Extended Kalman filter estimating the Thymio pose from odometry and camera.

    prediction : wheel speeds -> odometry motion model g        (every loop)
    update     : camera pose  -> direct measurement of the pose (when the robot is seen)

Course (MICRO-452):
    "Localisation" slide 18        odometry of a differential drive robot
    "Uncertainties" slides 42-46   Kalman filter equations
    "Uncertainties" slides 47-48   Extended Kalman filter (non-linear motion model)
    Exercise 8 solution            noise variances measured on recorded data

Units: x, y [mm], theta [rad] counter-clockwise from the x axis, speeds [mm/s], time [s].
"""

import numpy as np


def wrap_angle(angle):
    """Bring an angle back into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ExtendedKalmanFilter:
    """Pose mean mu = [x, y, theta] and covariance Sigma."""

    def __init__(self, initial_pose, initial_covariance, wheel_distance,
                 speed_variance, angular_speed_variance, camera_variances):
        self.pose = np.array(initial_pose, dtype=float)              # mu
        self.covariance = np.array(initial_covariance, dtype=float)  # Sigma
        self.wheel_distance = wheel_distance                         # b [mm]
        self.speed_variance = speed_variance                         # of v     [mm^2/s^2]
        self.angular_speed_variance = angular_speed_variance         # of omega [rad^2/s^2]
        self.camera_noise = np.diag(camera_variances)                # R
        self.measurement_matrix = np.eye(3)                          # H: the camera measures x, y, theta directly

    def predict(self, left_speed, right_speed, dt):
        """Prediction with the measured wheel speeds [mm/s] over dt [s]."""
        distance = (left_speed + right_speed) / 2 * dt                     # delta_s
        rotation = (right_speed - left_speed) / self.wheel_distance * dt  # delta_theta
        heading = self.pose[2] + rotation / 2                             # theta + delta_theta / 2

        # mu_bar = g(u, mu)
        self.pose = self.pose + np.array([distance * np.cos(heading),
                                          distance * np.sin(heading),
                                          rotation])
        self.pose[2] = wrap_angle(self.pose[2])

        # G = dg/dmu, linearisation of g around the previous pose
        motion_jacobian = np.array([[1.0, 0.0, -distance * np.sin(heading)],
                                    [0.0, 1.0, distance * np.cos(heading)],
                                    [0.0, 0.0, 1.0]])
        # Q: a speed error integrated over dt gives a pose error of variance sigma^2 * dt^2
        motion_noise = np.diag([self.speed_variance,
                                self.speed_variance,
                                self.angular_speed_variance]) * dt**2
        # Sigma_bar = G Sigma G^T + Q
        self.covariance = motion_jacobian @ self.covariance @ motion_jacobian.T + motion_noise

    def update(self, measured_pose):
        """Correction with a camera measurement z = [x, y, theta]. Returns the gain K for display."""
        H = self.measurement_matrix
        innovation = np.asarray(measured_pose, dtype=float) - H @ self.pose          # i = z - H mu_bar
        innovation[2] = wrap_angle(innovation[2])                                     # 179 deg vs -179 deg = 2 deg
        innovation_covariance = H @ self.covariance @ H.T + self.camera_noise         # S = H Sigma_bar H^T + R
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(innovation_covariance)    # K = Sigma_bar H^T S^-1

        self.pose = self.pose + kalman_gain @ innovation                              # mu = mu_bar + K i
        self.pose[2] = wrap_angle(self.pose[2])
        self.covariance = (np.eye(3) - kalman_gain @ H) @ self.covariance             # Sigma = (I - K H) Sigma_bar
        return kalman_gain
