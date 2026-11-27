import numpy as np

# Filter parameters (units: mm, s, rad)
WHEEL_DISTANCE = 95.0                                 # b, distance between the wheels (Thymio cheat sheet)
SPEED_VARIANCE = 1.55                                 # var(v) [mm^2/s^2], measured on our robot (Exercise 8 method)
ANGULAR_SPEED_VARIANCE = 9.06e-4                      # var(omega) [rad^2/s^2], measured on our robot
CAMERA_VARIANCES = [1.0, 1.0, np.radians(1.0) ** 2]  # R: var(x), var(y) [mm^2], var(theta) [rad^2], assumed


def wrap_angle(angle):
    """Bring an angle back into [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ExtendedKalmanFilter:
    """Pose mean mu = [x, y, theta] and covariance Sigma, started from a camera measurement."""

    def __init__(self, initial_pose):
        self.pose = np.array(initial_pose, dtype=float)  # mu
        self.covariance = np.diag(CAMERA_VARIANCES)      # Sigma: as uncertain as the camera at the start
        self.camera_noise = np.diag(CAMERA_VARIANCES)    # R
        self.measurement_matrix = np.eye(3)              # H: the camera measures x, y, theta directly

    def predict(self, left_speed, right_speed, dt):
        """Prediction with the measured wheel speeds [mm/s] over dt [s]."""
        distance = (left_speed + right_speed) / 2 * dt               # delta_s
        rotation = (right_speed - left_speed) / WHEEL_DISTANCE * dt  # delta_theta
        heading = self.pose[2] + rotation / 2                        # theta + delta_theta / 2

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
        motion_noise = np.diag([SPEED_VARIANCE, SPEED_VARIANCE, ANGULAR_SPEED_VARIANCE]) * dt**2
        # Sigma_bar = G Sigma G^T + Q
        self.covariance = motion_jacobian @ self.covariance @ motion_jacobian.T + motion_noise

    def update(self, measured_pose):
        """Correction with a camera measurement z = [x, y, theta]."""
        H = self.measurement_matrix
        innovation = np.asarray(measured_pose, dtype=float) - H @ self.pose          # i = z - H mu_bar
        innovation[2] = wrap_angle(innovation[2])                                     # 179 deg vs -179 deg = 2 deg
        innovation_covariance = H @ self.covariance @ H.T + self.camera_noise         # S = H Sigma_bar H^T + R
        kalman_gain = self.covariance @ H.T @ np.linalg.inv(innovation_covariance)    # K = Sigma_bar H^T S^-1

        self.pose = self.pose + kalman_gain @ innovation                              # mu = mu_bar + K i
        self.pose[2] = wrap_angle(self.pose[2])
        self.covariance = (np.eye(3) - kalman_gain @ H) @ self.covariance             # Sigma = (I - K H) Sigma_bar
