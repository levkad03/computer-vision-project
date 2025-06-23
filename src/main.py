import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from goal import (
    PostTracker,
    detect_goal_in_frame,
    draw_black_goal,
)
from pitch import detect_pitch, overlay_pitch_info, update_pitch_corners


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
