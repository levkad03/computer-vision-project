from typing import Optional, Tuple

import cv2
import numpy as np

from config import GOAL_CONFIG


def get_bright_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask for pixels within the bright HSV range.
    """
    lower_bright, upper_bright = GOAL_CONFIG.bright_hsv_range
    return cv2.inRange(hsv_frame, lower_bright, upper_bright)


def detect_two_peaks(bgr_strip: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    From a given vertical strip (in BGR) of the pitch, finds the x positions
    of two bright peaks (goal posts).
    Returns (left_peak, right_peak) or None if no valid peaks are found.
    """
    hsv_strip = cv2.cvtColor(bgr_strip, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(
        get_bright_mask(hsv_strip),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    energy = np.abs(cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=0)
    energy = cv2.blur(energy[:, None], (GOAL_CONFIG.sobel_blur_kernel, 1)).ravel()
    if energy.max() < 1:
        return None

    total_width = energy.size
    left_zone = int(GOAL_CONFIG.zone_fraction * total_width)
    right_zone = int((1 - GOAL_CONFIG.zone_fraction) * total_width)
    left_peak = np.argmax(energy[:left_zone])
    right_peak = np.argmax(energy[right_zone:]) + right_zone

    threshold = GOAL_CONFIG.min_peak_intensity_ratio * energy.max()
    if (
        energy[left_peak] < threshold
        or energy[right_peak] < threshold
        or (right_peak - left_peak) < GOAL_CONFIG.min_peak_gap
    ):
        return None

    return int(left_peak), int(right_peak)


def is_bright_confirmed(hsv_strip: np.ndarray, x: int) -> bool:
    """
    Checks if a narrow vertical slice around x in hsv_strip has enough bright pixels.
    """
    w = hsv_strip.shape[1]
    x0, x1 = (
        max(0, x - GOAL_CONFIG.recheck_radius),
        min(w, x + GOAL_CONFIG.recheck_radius + 1),
    )
    brightness_ratio = (get_bright_mask(hsv_strip[:, x0:x1]) > 0).mean()
    return brightness_ratio >= GOAL_CONFIG.recheck_threshold


class PostTracker:
    """
    Uses an exponential moving average to keep
    the goal post positions stable across frames.
    """

    def __init__(self) -> None:
        self.last_posts: Optional[Tuple[int, int]] = None

    def update(
        self, measurement: Optional[Tuple[int, int]], hsv_strip: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        if measurement:
            left, right = measurement
            if not (
                is_bright_confirmed(hsv_strip, left)
                and is_bright_confirmed(hsv_strip, right)
            ):
                measurement = None

        if self.last_posts and measurement:
            if (
                max(
                    abs(measurement[0] - self.last_posts[0]),
                    abs(measurement[1] - self.last_posts[1]),
                )
                > GOAL_CONFIG.max_allowed_jump
            ):
                measurement = None

        if measurement:
            if self.last_posts:
                smoothed_left = int(
                    GOAL_CONFIG.smoothing_alpha * measurement[0]
                    + (1 - GOAL_CONFIG.smoothing_alpha) * self.last_posts[0]
                )
                smoothed_right = int(
                    GOAL_CONFIG.smoothing_alpha * measurement[1]
                    + (1 - GOAL_CONFIG.smoothing_alpha) * self.last_posts[1]
                )
                self.last_posts = (smoothed_left, smoothed_right)
            else:
                self.last_posts = measurement
        return self.last_posts


def draw_black_goal(
    frame: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Detects and draws a fixed black-area (goal) on the given frame.
    Returns the top-left and bottom-right coordinates of the detected goal,
    or None if not found.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 40))
    black_mask = cv2.morphologyEx(
        black_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
    )
    cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        y += 50
        if not ((w * 0.05) * (h * 0.015) < area < (w * 0.15) * (h * 0.045)):
            continue
        ar = bw / bh
        if (
            not (2.5 < ar < 6.0)
            or not (y + bh / 2 < h * 0.25)
            or not (w * 0.3 < x + bw / 2 < w * 0.7)
        ):
            continue
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
        return (x, y), (x + bw, y + bh)
    return None


def detect_goal_in_frame(
    frame: np.ndarray, pitch_corners: np.ndarray, tracker: PostTracker
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Using the lower part (strip) of the pitch, detects goal posts
    and returns a goal rectangle.
    """
    poly = pitch_corners.astype(int).reshape(-1, 2)
    xL, xR = poly[:, 0].min(), poly[:, 0].max()
    yT, yB = poly[:, 1].min(), poly[:, 1].max()

    strip_top = max(int(yB - GOAL_CONFIG.strip_height), 0)
    strip_bottom = int(yB)
    strip_bgr = frame[strip_top:strip_bottom, xL:xR]
    strip_hsv = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2HSV)

    measurement = detect_two_peaks(strip_bgr)
    posts = tracker.update(measurement, strip_hsv)

    if posts:
        left_post, right_post = posts
        goal_tl = (xL + left_post, strip_top + 20)
        goal_br = (xL + right_post, yB)
        cv2.rectangle(frame, goal_tl, goal_br, (255, 0, 255), 2)
        return goal_tl, goal_br
    return None
