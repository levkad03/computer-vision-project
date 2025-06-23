from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ball import BallTracker
from config import GameState, VideoConfig


class VideoRenderer:
    """Handles video rendering and display."""

    def __init__(self, dest_path: Optional[Path], config: VideoConfig):
        self.writer = None

        if dest_path:
            fourcc = cv2.VideoWriter_fourcc(*config.fourcc)
            self.writer = cv2.VideoWriter(
                str(dest_path), fourcc, config.fps, config.warp_dims
            )

        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 960, 540)

    def draw_ui_elements(
        self,
        frame: np.ndarray,
        game_state: GameState,
        bottom_gate_coords,
        ball_center: Optional[Tuple[int, int]],
        ball_tracker: BallTracker,
    ):
        """Draw all UI elements on the frame."""
        # Goal detection indicator
        if bottom_gate_coords:
            cv2.putText(
                frame,
                "Goal Detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        # Ball trajectory
        ball_tracker.draw_trajectory(frame)

        # Score display
        score_text = f"Player 1 - {game_state.player1_score} | Player 2 - {game_state.player2_score}"
        text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 30
        cv2.putText(
            frame,
            score_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            2,
        )

    def display_and_write(self, frame: np.ndarray) -> bool:
        """Display frame and write to video file.
        Returns False if user wants to quit."""
        cv2.imshow("Detection", frame)
        if self.writer:
            self.writer.write(frame)
        return not (cv2.waitKey(1) & 0xFF == 27)

    def cleanup(self):
        """Clean up video resources."""
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
