import time
from pathlib import Path
from typing import Optional

import cv2

from ball import BallTracker
from config import GameState, VideoConfig
from goal import GoalDetector
from pitch import PitchTracker
from video import VideoRenderer


def process_video_with_yolo_and_goals(
    source_path: Path,
    dest_path: Optional[Path] = None,
    config: Optional[VideoConfig] = None,
) -> None:
    """
    Process foosball video with YOLO ball detection and goal tracking.

    Args:
        source_path: Path to input video
        dest_path: Path for output video (optional)
        config: Video processing configuration
    """

    if config is None:
        config = VideoConfig()

    # Initialize video capture
    cap = cv2.VideoCapture(str(source_path))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")

    # Initialize components
    ball_tracker = BallTracker("../weights/best.pt")
    pitch_tracker = PitchTracker(config.warp_dims)
    goal_detector = GoalDetector()
    game_state = GameState()
    renderer = VideoRenderer(dest_path, config)

    # Initialize pitch tracking
    try:
        warped = pitch_tracker.initialize(frame)
    except RuntimeError as e:
        cap.release()
        renderer.cleanup()
        raise e

    # Main processing loop
    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Update pitch and detect goals
            warped, bottom_gate_coords, upper_gate_coords = pitch_tracker.update(
                frame, frame_idx
            )

            # Detect ball
            ball_center = ball_tracker.detect_and_draw_ball_detection(warped, config)

            # Update game state
            game_state.update_cooldown()
            goal_detector.update_goal_positions(
                game_state, ball_center, bottom_gate_coords, upper_gate_coords
            )
            goal_detector.check_for_goals(game_state)

            # Render frame
            renderer.draw_ui_elements(
                warped, game_state, bottom_gate_coords, ball_center, ball_tracker
            )

            # Display and check for quit
            if not renderer.display_and_write(warped):
                break

            frame_idx += 1
    finally:
        # Cleanup
        cap.release()
        renderer.cleanup()

        # Print statistics
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"[DONE] {frame_idx} frames processed at {fps:.1f} FPS")
        print(
            f"Final Score - Player 1: {game_state.player1_score}, Player 2: {game_state.player2_score}"
        )


if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = VideoConfig(conf_threshold=0.4, iou_threshold=0.3, fps=30)

    process_video_with_yolo_and_goals(
        Path("data/Match.mp4"), Path("data/combined_output2.mp4"), custom_config
    )
