import time
import cv2
import numpy as np
from tdmclient import ClientAsync, aw

import vision as vis
from kalman import ExtendedKalmanFilter
from nav import global_navigation
from local_nav import Local, read_proximity
from motion_control import PathFollower, twist_to_wheels, wheels_to_targets


MAP_SIZE = 1000
RES = 10.0

R_ROBOT = 100.0
MARGIN = 20.0

CAMERA_INDEX = 1

SPEED_TO_MM_S = 0.3482
MARKERS_AHEAD_OF_AXLE_MM = 0.0

PROXIMITY = {
    "stop_raw": None,
    "start_avoiding_raw": None,
    "clear_raw": None,
    "no_wall_raw": None,
    "desired_wall_raw": None,
    "wall_gain_raw": None,
}


ROBOT_PROGRAM = """
var command_age = 0
var prox_tick = 0

motor.left.target = 0
motor.right.target = 0
timer.period[0] = 100

onevent speed
    motor.left.target = event.args[0]
    motor.right.target = event.args[1]
    command_age = 0

onevent timer0
    if command_age < 5 then
        command_age += 1
    end
    if command_age >= 5 then
        motor.left.target = 0
        motor.right.target = 0
    end

onevent prox
    if prox_tick >= 30000 then
        prox_tick = 0
    else
        prox_tick += 1
    end
"""


def select_points(image, count, title):

    points = []

    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse)

    try:
        while len(points) < count:
            preview = image.copy()

            for i, point in enumerate(points):
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
                cv2.putText(preview, str(i + 1), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(title, preview)

            if cv2.waitKey(20) & 0xFF == 27:
                raise KeyboardInterrupt

    finally:
        cv2.destroyWindow(title)

    return np.array(points, dtype=np.float32)


def prepare_map(frame):

    corners = select_points(frame, 4, "Corners: top-left, top-right, bottom-right, bottom-left")

    dst = np.float32([
        [-0.5, -0.5],
        [MAP_SIZE - 0.5, -0.5],
        [MAP_SIZE - 0.5, MAP_SIZE - 0.5],
        [-0.5, MAP_SIZE - 0.5],
    ])

    transform = cv2.getPerspectiveTransform(corners, dst)
    image = cv2.warpPerspective(frame, transform, (MAP_SIZE, MAP_SIZE))

    walls = vis.detect_walls(image)

    contours, _ = cv2.findContours(walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(walls, contours, -1, 255, cv2.FILLED)

    cells = int(MAP_SIZE / RES)
    cell_px = int(RES)

    grid = walls.reshape(cells, cell_px, cells, cell_px).max(axis=(1, 3)) > 0
    obstacle_map = np.flipud(grid).astype(np.uint8).copy()

    preview = image.copy()
    preview[walls > 0] = (0, 0, 255)

    u, v = select_points(preview, 1, "Check red obstacle mask, then click goal")[0]
    goal = np.array([u + 0.5, MAP_SIZE - v - 0.5])

    return obstacle_map, transform, goal


def camera_pose(image):

    coordinates = vis.detect_thymio(image)

    if coordinates is None:
        return None

    (u, v), theta_image = coordinates
    theta = -theta_image

    x = u + 0.5 - MARKERS_AHEAD_OF_AXLE_MM * np.cos(theta)
    y = MAP_SIZE - v - 0.5 - MARKERS_AHEAD_OF_AXLE_MM * np.sin(theta)

    return np.array([x, y, theta])


def path_info(points, pose):

    path = np.asarray(points, dtype=float)

    if len(path) == 1:
        return float(np.linalg.norm(pose[:2] - path[0])), 0.0, float(pose[2]), 0

    segments = path[1:] - path[:-1]
    lengths = np.linalg.norm(segments, axis=1)

    t = np.sum((pose[:2] - path[:-1]) * segments, axis=1) / np.maximum(lengths ** 2, 1e-12)
    t = np.clip(t, 0.0, 1.0)

    projections = path[:-1] + t[:, None] * segments
    distances = np.linalg.norm(projections - pose[:2], axis=1)

    i = int(np.argmin(distances))

    progress = np.sum(lengths[:i]) + t[i] * lengths[i]
    heading = np.arctan2(segments[i, 1], segments[i, 0])

    return float(distances[i]), float(progress), float(heading), i


def to_pixel(point):

    return int(round(point[0] - 0.5)), int(round(MAP_SIZE - point[1] - 0.5))


def send_speed(node, v, w):

    if not np.all(np.isfinite([v, w])):
        raise ValueError("Neispravna komanda brzine")

    left, right = wheels_to_targets(*twist_to_wheels(v, w))

    aw(node.send_events({"speed": [left, right]}))


def main():

    if any(value is None for value in PROXIMITY.values()):
        raise ValueError("Prvo popuni izmjerene pragove i wall_gain_raw u PROXIMITY")

    local = Local(**PROXIMITY)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        if not cap.isOpened():
            raise RuntimeError("Kamera nije otvorena")

        with ClientAsync() as client:

            node = aw(client.wait_for_node())
            aw(node.lock())

            try:
                aw(node.register_events([("speed", 2)]))

                error = aw(node.compile(ROBOT_PROGRAM))
                if error is not None:
                    raise RuntimeError(f"Aseba compile: {error}")

                error = aw(node.run())
                if error is not None:
                    raise RuntimeError(f"Aseba run: {error}")

                aw(node.wait_for_variables({
                    "prox.horizontal",
                    "motor.left.speed",
                    "motor.right.speed",
                    "prox_tick",
                }))

                ok, frame = cap.read()

                if not ok:
                    raise RuntimeError("Nema slike sa kamere")

                obstacle_map, transform, goal = prepare_map(frame)

                ekf = None
                follower = None
                started = False

                previous_time = time.monotonic()
                last_camera_time = previous_time
                last_prox_time = previous_time

                last_tick = None
                last_status = None

                while True:

                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError("Prekinut je prijem slike")

                    now = time.monotonic()
                    dt = now - previous_time
                    previous_time = now

                    if started and not 0 < dt <= 0.3:
                        raise RuntimeError("Kontrolna petlja je prespora")

                    front_raw, right_raw, prox = read_proximity(client, node)

                    tick = node.v.prox_tick

                    if tick != last_tick:
                        last_tick = tick
                        last_prox_time = now

                    if now - last_prox_time > 0.3:
                        raise RuntimeError("Nema novih proximity podataka")

                    left_speed = node.v.motor.left.speed * SPEED_TO_MM_S
                    right_speed = node.v.motor.right.speed * SPEED_TO_MM_S

                    image = cv2.warpPerspective(frame, transform, (MAP_SIZE, MAP_SIZE))
                    measured_pose = camera_pose(image)

                    if ekf is not None:
                        ekf.predict(left_speed, right_speed, dt)

                    if measured_pose is not None:
                        last_camera_time = now

                        if ekf is None:
                            ekf = ExtendedKalmanFilter(measured_pose)
                        else:
                            ekf.update(measured_pose)

                    if started and now - last_camera_time > 1.0:
                        raise RuntimeError("Robot predugo nije vidljiv kameri")

                    v, w = 0.0, 0.0
                    status = "P: plan | S: start | ESC: stop"

                    if ekf is not None:

                        pose = ekf.pose.copy()

                        if not np.all(np.isfinite(pose)):
                            raise RuntimeError("Neispravna procjena polozaja")

                        if started:

                            dist, progress, path_dir, index = path_info(follower.points, pose)

                            follower.closest = index
                            vt, wt = follower.step(pose)

                            if follower.finished:
                                print("GOAL REACHED")
                                break

                            if max(prox) >= local.stop_raw:
                                local.stop("TOO_CLOSE_ANY_SENSOR")

                            v, w, status = local.step(
                                front_raw,
                                right_raw,
                                vt,
                                wt,
                                dt,
                                path_dir,
                                pose[2],
                                dist,
                                progress,
                            )

                            if local.mode == "STOP":
                                print(status)
                                break

                        center = to_pixel(pose[:2])

                        tip_position = pose[:2] + 50 * np.array([
                            np.cos(pose[2]),
                            np.sin(pose[2]),
                        ])

                        tip = to_pixel(tip_position)

                        cv2.arrowedLine(image, center, tip, (255, 0, 0), 2)

                    send_speed(node, v, w)

                    if follower is not None:

                        path_px = np.array([
                            to_pixel(p) for p in follower.points
                        ], dtype=np.int32)

                        cv2.polylines(image, [path_px.reshape(-1, 1, 2)], False, (255, 255, 0), 2)

                    cv2.circle(image, to_pixel(goal), 7, (0, 0, 255), -1)

                    cv2.putText(image, status, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(image, f"front {front_raw}   right {right_raw}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    cv2.imshow("Navigation", image)

                    if status != last_status:
                        print(status)
                        last_status = status

                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        break

                    if key == ord("p") and follower is None and measured_pose is not None:

                        path = global_navigation(
                            obstacle_map,
                            tuple(ekf.pose[:2]),
                            tuple(goal),
                            RES,
                            R_ROBOT,
                            MARGIN,
                        )

                        if not path:
                            raise RuntimeError("Nema prohodne globalne putanje")

                        follower = PathFollower(
                            path,
                            v_max=local.vmax,
                            omega_max=local.wmax,
                        )

                        previous_time = time.monotonic()

                    if key == ord("s") and follower is not None and not started and measured_pose is not None:

                        start_distance = np.linalg.norm(
                            ekf.pose[:2] - np.asarray(follower.points[0])
                        )

                        if start_distance > 30.0:
                            raise RuntimeError("Robot je pomjeren nakon planiranja; ponovo pokreni test")

                        local.reset()
                        started = True

                    aw(client.sleep(0.03))

            finally:
                try:
                    aw(node.set_variables({
                        "motor.left.target": [0],
                        "motor.right.target": [0],
                    }))
                finally:
                    aw(node.unlock())

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()