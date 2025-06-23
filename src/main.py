import time
from pathlib import Path
from dataclasses import dataclass
from typing import Final, Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PitchConfig:
    green_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (35, 40, 40),
        (85, 255, 255),
    )
    redetection_interval: int = 30
    max_corner_displacement: float = 6.0


@dataclass(frozen=True)
class GoalPostConfig:
    strip_height: int = 80
    bright_hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 200),
        (180, 60, 255),
    )
    sobel_blur_kernel: int = 21
    min_peak_intensity_ratio: float = 0.30
    min_peak_gap: int = 80
    zone_fraction: float = 0.4
    max_allowed_jump: int = 30
    recheck_radius: int = 3
    recheck_threshold: float = 0.40
    smoothing_alpha: float = 0.35


LABEL_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
PITCH_CONFIG = PitchConfig()
GOAL_CONFIG = GoalPostConfig()


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


def get_bright_mask(hsv_frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask for pixels within the bright HSV range.
    """
    lower_bright, upper_bright = GOAL_CONFIG.bright_hsv_range
    return cv2.inRange(hsv_frame, lower_bright, upper_bright)


def detect_two_peaks(bgr_strip: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    From a given vertical strip (in BGR) of the pitch, finds the x positions of two bright peaks (goal posts).
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
    Uses an exponential moving average to keep the goal post positions stable across frames.
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
    Returns the top-left and bottom-right coordinates of the detected goal, or None if not found.
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


def overlay_pitch_info(frame: np.ndarray, corners: np.ndarray) -> None:
    """
    Overlays the pitch area (polygon and corner labels) onto the frame.
    """
    poly = corners.astype(int).reshape(-1, 2)
    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
    labels = ("TL", "TR", "BR", "BL")
    for label, (x, y) in zip(labels, poly):
        cv2.putText(frame, label, (x + 5, y - 5), LABEL_FONT, 0.5, (0, 255, 0), 1)


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


def detect_goal_in_frame(
    frame: np.ndarray, pitch_corners: np.ndarray, tracker: PostTracker
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Using the lower part (strip) of the pitch, detects goal posts and returns a goal rectangle.
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


def process_video_with_yolo_and_goals(
    source_path: Path, dest_path: Optional[Path] = None
) -> None:
    cap = cv2.VideoCapture(str(source_path))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")
    height, width = frame.shape[:2]

    warp_dims = (1080, 1920)

    writer = None
    if dest_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest_path), fourcc, 30, warp_dims)

    ball_model = YOLO("../weights/best.pt")

    H = np.eye(3, dtype=np.float32)

    last_valid_positions: List[Tuple[int, int]] = []
    max_positions = 50

    warped = cv2.warpPerspective(frame, H, warp_dims)
    initial_hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    init_corners = detect_pitch(initial_hsv)
    if init_corners is None:
        raise RuntimeError("Pitch not detected in first frame")
    pitch_corners = init_corners.reshape(-1, 1, 2).astype(np.float32)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    prev_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    post_tracker = PostTracker()

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection", 960, 540)

    frame_idx = 0
    start_time = time.time()
    goal_cooldown = 0  # Prevents double-scoring
    player1_score = 0  # Top player
    player2_score = 0  # Bottom player
    ball_inside_bottom_goal = False
    ball_inside_upper_goal = False
    missing_counter = 0
    max_missing_frames = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        warped = cv2.warpPerspective(frame, H, warp_dims)

        # ---- Pitch & Goal-Post Detection ----
        pitch_corners, current_gray = update_pitch_corners(
            warped, pitch_corners, prev_gray, frame_idx, lk_params
        )
        bottom_gate_coords = detect_goal_in_frame(warped, pitch_corners, post_tracker)
        if bottom_gate_coords:
            cv2.putText(
                warped,
                "Goal Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        upper_gate_coords = draw_black_goal(warped)
        overlay_pitch_info(warped, pitch_corners)

        # ---- YOLO Ball Detection ----
        results = ball_model.track(
            warped,
            persist=True,
            conf=0.3,  # Lower this if needed
            tracker="bytetrack.yaml",  # or "bytetrack.yaml"
            imgsz=640,  # Match your model's training size
            iou=0.3,  # Adjust intersection over union threshold
            verbose=False,
        )[0]
        if results and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw detected ball rectangle and confidence label.
                cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    warped,
                    f"Ball {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

                # Compute center and update trajectory.
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                last_valid_positions.append(center)
                if len(last_valid_positions) > max_positions:
                    last_valid_positions.pop(0)

                # Draw trajectory lines.
                for i in range(1, len(last_valid_positions)):
                    cv2.line(
                        warped,
                        last_valid_positions[i - 1],
                        last_valid_positions[i],
                        (0, 0, 255),
                        2,
                    )
                break  # Only process the first detected ball

        if goal_cooldown > 0:
            goal_cooldown -= 1

        if results and results.boxes is not None:
            missing_counter = 0
            cx, cy = last_valid_positions[-1]

            if bottom_gate_coords:
                x1, y1 = bottom_gate_coords[0]
                x2, y2 = bottom_gate_coords[1]
                if x1 < cx < x2 and y1 < cy < y2:
                    ball_inside_bottom_goal = True

            if upper_gate_coords:
                x1, y1 = upper_gate_coords[0]
                x2, y2 = upper_gate_coords[1]
                if x1 < cx < x2 and y1 < cy < y2:
                    ball_inside_upper_goal = True
        else:
            missing_counter += 1
            if missing_counter > max_missing_frames and goal_cooldown == 0:
                if ball_inside_bottom_goal:
                    player1_score += 1
                    print("⚽ Goal for Player 1!")
                    goal_cooldown = 10
                if ball_inside_upper_goal:
                    player2_score += 1
                    print("⚽ Goal for Player 2!")
                    goal_cooldown = 10

                ball_inside_bottom_goal = False
                ball_inside_upper_goal = False
                missing_counter = 0

        score_text = f"Player 1 - {player1_score} | Player 2 - {player2_score}"
        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        text_x = (warped.shape[1] - text_size[0]) // 2
        text_y = warped.shape[0] - 30
        cv2.putText(
            warped,
            score_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Detection", warped)
        if writer:
            writer.write(warped)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        prev_gray = current_gray
        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"[DONE] {frame_idx} frames processed at {fps:.1f} FPS")


if __name__ == "__main__":
    process_video_with_yolo_and_goals(
        Path("data/Match.mp4"), Path("data/combined_output2.mp4")
    )
