import numpy as np


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


class Local:

    def __init__(self):

        self.mode = "FOLLOW_PATH"
        self.path_progress = 0.0
        self.stop_reason = None
        self.turning_left = False

        self.stop_dist = 35.0
        self.start_avoiding_dist = 80.0
        self.safe_dist = 120.0

        self.vmax = 50.0
        self.wmax = 0.7
        self.vturn = 10.0
        self.vfw = 30.0
        self.wturn = 0.45

        self.progress_before = 0.0
        self.avoiding_time = 0.0
        self.avoiding_time_treshold = 40.0
        self.wall_search_time = 0.0
        self.wall_search_time_treshold = 2.0

        self.no_wall_dist = 120.0
        self.min_wall_dist = 70.0

        self.rejoin_time = 0.0
        self.rejoin_dist = 30.0
        self.required_rejoin_progress = 100.0
        self.rejoin_angle_tolerance = np.radians(25)
        self.rejoin_time_treshold = 0.3

    def stop(self, reason):
        self.mode = "STOP"
        self.stop_reason = reason

        return 0.0, 0.0, reason

    def check_stop(self, front_dist, right_dist):

        if self.mode == "STOP":
            return 0.0, 0.0, self.stop_reason

        if min(front_dist, right_dist) <= self.stop_dist:
            return self.stop("TOO_CLOSE")

        return None

    def check_rejoin(self, dist_from_path, front_dist, err):

        a = dist_from_path <= self.rejoin_dist
        b = self.path_progress >= self.progress_before + self.required_rejoin_progress
        c = abs(err) <= self.rejoin_angle_tolerance
        d = front_dist >= self.safe_dist

        return a and b and c and d

    def step(self, front_dist, right_dist, vt, wt, dt, path_dir, robot_dir, dist_from_path, path_progress):

        stop_command = self.check_stop(front_dist, right_dist)

        if stop_command is not None:
            return stop_command

        self.path_progress = path_progress

        if self.mode == "FOLLOW_PATH":

            if front_dist > self.start_avoiding_dist:
                ff = (front_dist - self.start_avoiding_dist) / (self.safe_dist - self.start_avoiding_dist)
                ff = np.clip(ff, 0, 1)
                v = ff * np.clip(vt, 0, self.vmax)
                w = np.clip(wt, -self.wmax, self.wmax)

                return v, w, "FOLLOW_PATH"

            else:
                self.mode = "AVOID"
                self.turning_left = True
                self.progress_before = self.path_progress
                self.avoiding_time = 0.0
                self.rejoin_time = 0.0
                self.wall_search_time = 0.0

                return 0.0, 0.0, "OBSTACLE"

        elif self.mode == "AVOID":

            self.avoiding_time += dt

            if self.avoiding_time >= self.avoiding_time_treshold:
                return self.stop("AVOID_TIMEOUT")

            dir_err = wrap(path_dir - robot_dir)

            if self.check_rejoin(dist_from_path, front_dist, dir_err):
                self.rejoin_time += dt
            else:
                self.rejoin_time = 0.0

            if self.rejoin_time >= self.rejoin_time_treshold:
                self.mode = "FOLLOW_PATH"
                self.turning_left = False
                self.rejoin_time = 0.0
                self.wall_search_time = 0.0

                return 0.0, 0.0, "PATH JOINED"

            if front_dist <= self.start_avoiding_dist:
                self.turning_left = True
            elif front_dist >= self.safe_dist:
                self.turning_left = False

            if self.turning_left:
                self.wall_search_time = 0.0

                ff = (front_dist - self.stop_dist) / (self.start_avoiding_dist - self.stop_dist)
                ff = np.clip(ff, 0, 1)

                v = self.vturn * ff
                w = self.wturn

                return v, w, "AVOIDING LEFT"

            elif right_dist >= self.no_wall_dist:
                self.wall_search_time += dt

                if self.wall_search_time >= self.wall_search_time_treshold:
                    return self.stop("WALL_SEARCH")

                return self.vturn, -self.wturn, "GOING RIGHT"

            else:
                self.wall_search_time = 0.0

                err = self.min_wall_dist - right_dist
                w = np.clip(err * 0.01, -self.wmax, self.wmax)
                wf = 1 - abs(w) / self.wmax * 0.6

                ff = (front_dist - self.start_avoiding_dist) / (self.safe_dist - self.start_avoiding_dist)
                ff = np.clip(ff, 0, 1)

                v = self.vfw * ff * wf

                return v, w, "FOLLOW WALL"

        return self.stop("INVALID_MODE")

    def reset(self):
        self.mode = "FOLLOW_PATH"
        self.turning_left = False
        self.stop_reason = None
        self.path_progress = 0.0
        self.progress_before = 0.0
        self.avoiding_time = 0.0
        self.rejoin_time = 0.0
        self.wall_search_time = 0.0













































