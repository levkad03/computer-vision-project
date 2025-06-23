from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import VideoConfig


class BallTracker:
    """Handles ball detection and trajectory tracking."""

    def __init__(self, model_path: str, max_positions: int = 50):
        self.model = YOLO(model_path)
        self.last_valid_positions: List[Tuple[int, int]] = []
        self.max_positions = max_positions

    def detect_and_draw_ball_detection(
        self, frame: np.ndarray, config: VideoConfig
    ) -> Optional[Tuple[int, int]]:
        """Detect ball in frame and return center coordinates."""

        results = self.model.track(
            frame,
            persist=True,
            conf=config.conf_threshold,
            tracker=config.tracker,
            imgsz=config.imgsz,
            iou=config.iou_threshold,
            verbose=False,
        )[0]

        if results and results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Store detection info for drawing
                self.last_detection_info = {
                    "center": center,
                    "confidence": conf,
                    "box_coords": (x1, y1, x2, y2),
                }

                # Draw detection immediately
                self.draw_detection(frame, center, conf, (x1, y1, x2, y2))

                # Update trajectory
                self.last_valid_positions.append(center)
                if len(self.last_valid_positions) > self.max_positions:
                    self.last_valid_positions.pop(0)

                return center

        # Clear detection info if no ball found
        self.last_detection_info = None
        return None

    def draw_trajectory(self, frame: np.ndarray):
        """Draw ball trajectory lines."""
        for i in range(1, len(self.last_valid_positions)):
            cv2.line(
                frame,
                self.last_valid_positions[i - 1],
                self.last_valid_positions[i],
                (0, 0, 255),
                2,
            )
