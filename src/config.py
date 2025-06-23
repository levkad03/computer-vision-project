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


LABEL_FONT: Final = cv2.FONT_HERSHEY_SIMPLEX
PITCH_CONFIG = PitchConfig()
GOAL_CONFIG = GoalPostConfig()
