from dataclasses import dataclass
from typing import Final, Tuple

import cv2


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


@dataclass
class VideoConfig:
    """Configuration for video processing."""

    warp_dims: Tuple[int, int] = (1080, 1920)
    fps: int = 30
    fourcc: str = "mp4v"
    conf_threshold: float = 0.3
    iou_threshold: float = 0.3
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"


@dataclass
class GameState:
    """Tracks the current state of the foosball game."""

    player1_score: int = 0
    player2_score: int = 0
    goal_cooldown: int = 0
    ball_inside_bottom_goal: bool = False
    ball_inside_upper_goal: bool = False
    missing_counter: int = 0
    max_missing_frames: int = 5

    def update_cooldown(self):
        """Decrease goal cooldown if active."""
        if self.goal_cooldown > 0:
            self.goal_cooldown -= 1

    def reset_ball_positions(self):
        """Reset ball position tracking."""
        self.ball_inside_bottom_goal = False
        self.ball_inside_upper_goal = False
        self.missing_counter = 0

    def score_goal(self, player: int, cooldown_frames: int = 10):
        """Score a goal for the specified player."""

        if player == 1:
            self.player1_score += 1
            print("⚽ Goal for Player 1!")
        elif player == 2:
            self.player2_score += 1
            print("⚽ Goal for Player 2!")

        self.goal_cooldown = cooldown_frames
        self.reset_ball_positions()


LABEL_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
PITCH_CONFIG = PitchConfig()
GOAL_CONFIG = GoalPostConfig()
