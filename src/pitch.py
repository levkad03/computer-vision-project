from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import LABEL_FONT, PITCH_CONFIG
from goal import PostTracker, detect_goal_in_frame, draw_black_goal


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Orders the input points into [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    s = points.sum(axis=1)
    d = np.diff(points, axis=1)
    ordered = np.array(
        [
            points[np.argmin(s)],
            points[np.argmin(d)],
            points[np.argmax(s)],
            points[np.argmax(d)],
        ],
        dtype=np.float32,
    )
    return ordered


def detect_pitch(hsv_frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detects the green pitch region in an HSV image.
    Returns a 4x2 array of the pitch corner points (ordered TL, TR, BR, BL) or None.
    """
    lower_green, upper_green = PITCH_CONFIG.green_range
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50_000:
        return None

    hull = cv2.convexHull(largest)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) != 4:
        return None

    return order_corners(approx.reshape(-1, 2))


def update_pitch_corners(
    frame: np.ndarray,
    current_corners: np.ndarray,
    prev_gray: np.ndarray,
    frame_idx: int,
    lk_params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Either re-detects the pitch or updates its corners via optical flow.
    Returns the possibly updated corners and the current grayscale frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if frame_idx % PITCH_CONFIG.redetection_interval == 0:
        new_detection = detect_pitch(hsv)
        if new_detection is not None:
            current_corners = new_detection.reshape(-1, 1, 2).astype(np.float32)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, current_gray, current_corners, None, **lk_params
    )
    if (
        status.sum() == 4
        and np.max(np.linalg.norm(new_corners - current_corners, axis=2))
        < PITCH_CONFIG.max_corner_displacement
    ):
        current_corners = new_corners
    return current_corners, current_gray


def overlay_pitch_info(frame: np.ndarray, corners: np.ndarray) -> None:
    """
    Overlays the pitch area (polygon and corner labels) onto the frame.
    """
    poly = corners.astype(int).reshape(-1, 2)
    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, poly):
        cv2.putText(frame, label, (x + 5, y - 5), LABEL_FONT, 0.5, (0, 255, 0), 1)


class PitchTracker:
    """Handles pitch detection and tracking."""

    def __init__(self, warp_dims: Tuple[int, int]):
        self.warp_dims = warp_dims
        self.H = np.eye(3, dtype=np.float32)
        self.pitch_corners = None
        self.post_tracker = PostTracker()
        self.lk_parameters = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray = None

    def initialize(self, frame: np.ndarray) -> np.ndarray:
        """Initialize pitch tracking with first frame."""

        warped = cv2.warpPerspective(frame, self.H, self.warp_dims)
        initial_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        init_corners = detect_pitch(initial_hsv)

        if init_corners is None:
            raise RuntimeError("Pitch not detected in first frame")

        self.pitch_corners = init_corners.reshape(-1, 1, 2).astype(np.float32)
        self.prev_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return warped

    def update(
        self, frame: np.ndarray, frame_idx: int
    ) -> Tuple[np.ndarray, Optional[List], Optional[List]]:
        """Update pitch tracking and detect goals."""

        warped = cv2.warpPerspective(frame, self.H, self.warp_dims)
        self.pitch_corners, current_gray = update_pitch_corners(
            warped, self.pitch_corners, self.prev_gray, frame_idx, self.lk_parameters
        )

        bottom_gate_coords = detect_goal_in_frame(
            warped, self.pitch_corners, self.post_tracker
        )

        upper_gate_coords = draw_black_goal(warped)
        overlay_pitch_info(warped, self.pitch_corners)

        self.prev_gray = current_gray

        return warped, bottom_gate_coords, upper_gate_coords
