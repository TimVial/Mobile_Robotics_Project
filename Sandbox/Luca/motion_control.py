import math


# Robot parameters — calibrate on your robot.
WHEEL_BASE = 95.0       # Distance between wheels [mm]
SPEED_FACTOR = 0.3375  # mm/s per Thymio motor unit
MAX_TARGET = 500        # Maximum absolute motor command


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def twist_to_wheels(v, omega):
    """Robot velocity [mm/s, rad/s] -> wheel velocities [mm/s]."""
    v_left = v - 0.5 * WHEEL_BASE * omega
    v_right = v + 0.5 * WHEEL_BASE * omega
    return v_left, v_right


def wheels_to_twist(v_left, v_right):
    """Wheel velocities [mm/s] -> robot velocity [mm/s, rad/s]."""
    v = 0.5 * (v_left + v_right)
    omega = (v_right - v_left) / WHEEL_BASE
    return v, omega


def wheels_to_targets(v_left, v_right):
    """Wheel velocities [mm/s] -> integer Thymio motor commands."""
    left = v_left / SPEED_FACTOR
    right = v_right / SPEED_FACTOR

    # Scale both commands equally to preserve their ratio.
    scale = max(
        1.0,
        abs(left) / MAX_TARGET,
        abs(right) / MAX_TARGET,
    )

    return int(round(left / scale)), int(round(right / scale))


def targets_to_wheels(left, right):
    """Thymio motor units -> wheel velocities [mm/s]."""
    return left * SPEED_FACTOR, right * SPEED_FACTOR


def densify(path, step):
    """Add points so segment spacing is at most step [mm]."""
    points = [tuple(path[0])]

    for a, b in zip(path[:-1], path[1:]):
        n = max(1, math.ceil(math.dist(a, b) / step))

        points += [
            (
                a[0] + (b[0] - a[0]) * k / n,
                a[1] + (b[1] - a[1]) * k / n,
            )
            for k in range(1, n + 1)
        ]

    return points


class PathFollower:
    """Lookahead path follower with proportional heading control."""

    def __init__(
        self,
        path,
        v_max=80.0,
        k_alpha=2.0,
        omega_max=1.5,
        lookahead=80.0,
        k_rho=1.0,
        goal_damping=50.0,
        goal_tol=15.0,
        resolution=10.0,
        window=30,
    ):
        self.v_max = v_max
        self.k_alpha = k_alpha
        self.omega_max = omega_max
        self.lookahead = lookahead
        self.k_rho = k_rho
        self.goal_damping = goal_damping
        self.goal_tol = goal_tol
        self.resolution = resolution
        self.window = window

        self.set_path(path)

    def set_path(self, path):
        """Initialize or replace the global path."""
        self.points = densify(path, self.resolution) if path else []
        self.closest = 0
        self.target = 0
        self.finished = not self.points

    def step(self, pose):
        """pose = (x, y, theta) [mm, mm, rad]; returns (v, omega)."""
        if self.finished:
            return 0.0, 0.0

        x, y, theta = pose
        goal = self.points[-1]

        rho_goal = math.hypot(goal[0] - x, goal[1] - y)

        if rho_goal < self.goal_tol:
            self.finished = True
            return 0.0, 0.0

        # Search for the closest point in a forward index window.
        best = math.inf

        for i in range(
            self.closest,
            min(len(self.points), self.closest + self.window),
        ):
            distance = math.dist(self.points[i], (x, y))

            if distance < best:
                best = distance
                self.closest = i

        # Select the first forward point outside the lookahead radius.
        self.target = len(self.points) - 1

        for i in range(self.closest, len(self.points)):
            if math.dist(self.points[i], (x, y)) > self.lookahead:
                self.target = i
                break

        tx, ty = self.points[self.target]

        alpha = wrap_angle(math.atan2(ty - y, tx - x) - theta)

        omega = max(
            -self.omega_max,
            min(self.omega_max, self.k_alpha * alpha),
        )

        # No forward motion if the target is more than 90 degrees away.
        v = self.v_max * max(math.cos(alpha), 0.0)

        # Slow down near the final goal and reduce rotation.
        if self.target == len(self.points) - 1:
            v = min(v, self.k_rho * rho_goal)
            omega *= min(1.0, rho_goal / self.goal_damping)

        return v, omega


class MotionController:
    """Select path/local commands and convert them to motor targets."""

    def __init__(self, path, hold_steps=5, **follower_params):
        self.follower = PathFollower(path, **follower_params)

        self.hold_steps = hold_steps
        self.mode = "PATH"  # PATH, AVOID or DONE

        self._last_local = None
        self._hold = 0

    def step(self, pose, local_cmd=None):
        """
        pose: (x, y, theta) [mm, mm, rad]
        local_cmd: None or (v, omega) [mm/s, rad/s]

        Returns:
            (left, right) integer Thymio motor commands.
        """
        if local_cmd is not None:
            self._last_local = local_cmd
            self._hold = self.hold_steps

        elif self._hold > 0:
            self._hold -= 1
            local_cmd = self._last_local

        if local_cmd is not None:
            self.mode = "AVOID"
            v, omega = local_cmd

        else:
            v, omega = self.follower.step(pose)
            self.mode = "DONE" if self.follower.finished else "PATH"

        return wheels_to_targets(*twist_to_wheels(v, omega))