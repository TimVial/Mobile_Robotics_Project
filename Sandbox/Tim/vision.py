"""Pose of a white Thymio filmed from above on a coloured floor (e.g. by a phone), no calibration.

Two pixel coordinate systems:
    image (u, v): OpenCV convention, origin top-left, v downwards
    pose  (x, y): origin bottom-left, y upwards, angle counter-clockwise from x (0 = facing right)
The pose is given at the centre of the wheel axle, the point whose motion odometry describes.

Pipeline (course: Exercise 2 and OpenCV tutorial notebooks):
    HSV -> bright and colourless pixels -> opening -> largest robot-like external contour
Colour choice (course LDA criterion J, Perception slides 40-43, measured on Thymio_camera.png):
the saturation S separates the white robot from the beige floor far better than the gray level
(J = 17.5 against 7.5) and does not depend on the light intensity. The threshold is a fraction of
the floor saturation (median of the image), so it also follows the colour of the light.
"""

from typing import NamedTuple

import cv2
import numpy as np

AXLE_BEHIND_CENTRE = 0.22   # wheel axle (pen hole) lies 22 % of the robot size behind its centre
MIN_BRIGHTNESS = 60         # darker pixels have no reliable colour
MIN_AREA_FRACTION = 0.001   # smaller blobs are noise
MAX_ASPECT_RATIO = 1.3      # seen from above, the robot is almost square
MIN_FILL_RATIO = 0.8        # contour area / bounding box area (Thymio: ~0.95)


class RobotDetection(NamedTuple):
    x: float      # [px]
    y: float      # [px]
    angle: float  # [rad]
    size: float   # [px] mean side of the robot bounding box, gives the pixel -> mm scale


def open_camera(source):
    """Webcam index (0, 1, ...) or phone stream URL, e.g. 'http://192.168.1.12:4747/video'."""
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open the camera {source!r}")
    return camera


def read_frame(camera):
    is_frame_read, frame = camera.read()
    if not is_frame_read:
        raise RuntimeError("The camera stream stopped")
    return frame


def segment_white(frame, saturation_ratio):
    """Mask (255) of the bright pixels whose saturation is below saturation_ratio x the floor saturation.

    The floor fills most of the image, so the median saturation is the floor's one. With the measured
    means (robot 29, floor 154), the course discriminator (29 + 154) / 2 = 91 is 0.6 x the floor.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    max_saturation = saturation_ratio * np.median(hsv[:, :, 1])
    mask = cv2.inRange(hsv, (0, 0, MIN_BRIGHTNESS), (179, max_saturation, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # removes small specks


def looks_like_robot(contour, image_shape):
    """Large enough, almost square, well filled and not cut by the image border."""
    area = cv2.contourArea(contour)
    _, (box_width, box_height), _ = cv2.minAreaRect(contour)
    left, top, width, height = cv2.boundingRect(contour)
    image_height, image_width = image_shape[:2]
    is_cut_by_border = left == 0 or top == 0 or left + width == image_width or top + height == image_height
    return (not is_cut_by_border
            and area > MIN_AREA_FRACTION * image_height * image_width
            and max(box_width, box_height) < MAX_ASPECT_RATIO * min(box_width, box_height)
            and area > MIN_FILL_RATIO * box_width * box_height)


def detect_robot(frame, saturation_ratio):
    """RobotDetection of the robot, or None if no robot-like blob is visible."""
    mask = segment_white(frame, saturation_ratio)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    robot_like_contours = [contour for contour in contours if looks_like_robot(contour, mask.shape)]
    if not robot_like_contours:
        return None
    contour = max(robot_like_contours, key=cv2.contourArea)
    (box_u, box_v), (box_width, box_height), box_angle = cv2.minAreaRect(contour)

    # The rounded front holds less white than the square back, so the centroid lies behind
    # the box centre. The straight edges give the heading modulo 90 deg: keep the box axis
    # pointing from the centroid towards the box centre.
    moments = cv2.moments(contour)
    centroid = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
    box_centre = np.array([box_u, box_v])
    axis_angles = np.radians(box_angle) + np.arange(4) * np.pi / 2
    axis_directions = np.column_stack([np.cos(axis_angles), np.sin(axis_angles)])
    front = axis_directions[np.argmax(axis_directions @ (box_centre - centroid))]

    size = (box_width + box_height) / 2
    axle_u, axle_v = box_centre - AXLE_BEHIND_CENTRE * size * front
    return RobotDetection(x=axle_u,
                          y=frame.shape[0] - axle_v,
                          angle=np.arctan2(-front[1], front[0]),  # minus: v points down
                          size=size)


def draw_pose(frame, x, y, angle, color, length=40):
    """Arrow from the pose (x, y) [px] along its angle [rad]."""
    height = frame.shape[0]
    tail = (int(x), int(height - y))
    tip = (int(x + length * np.cos(angle)), int(height - y - length * np.sin(angle)))
    cv2.arrowedLine(frame, tail, tip, color, 2, tipLength=0.3)


def draw_covariance_ellipse(frame, x, y, covariance, color):
    """Ellipse holding 95 % of a 2D Gaussian centred on (x, y), covariance in px^2."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)                      # ascending order
    minor_axis, major_axis = 2.45 * np.sqrt(np.maximum(eigenvalues, 0))        # 2.45^2 = chi2(2 dof, 95 %)
    major_angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    centre = (int(x), int(frame.shape[0] - y))
    cv2.ellipse(frame, centre, (int(major_axis), int(minor_axis)), -major_angle, 0, 360, color, 1)
