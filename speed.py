"""Perspective-transform road coordinates and estimate track speed."""
from collections import defaultdict, deque

import cv2
import numpy as np

import config


def video_source(value):
    if isinstance(value, int):
        return value
    from pathlib import Path
    path = Path(value)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return str(path.resolve())


class SpeedEstimator:
    def __init__(self, fps, frame_size=None):
        self.fps = fps if fps > 0 else 30.0
        points = np.asarray(config.SOURCE_POINTS, dtype=np.float32)
        if frame_size and points.max() <= 1.0:
            width, height = frame_size
            points[:, 0] *= width
            points[:, 1] *= height
        destination = np.float32([
            [0, 0], [config.REAL_WIDTH_M, 0],
            [config.REAL_WIDTH_M, config.REAL_LENGTH_M], [0, config.REAL_LENGTH_M],
        ])
        self.matrix = cv2.getPerspectiveTransform(points, destination)
        self.polygon = points.astype(np.int32)
        self.history = defaultdict(lambda: deque(maxlen=config.SPEED_WINDOW))
        self.smooth_speed = {}

    def inside(self, x, y):
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def project(self, x, y):
        if not self.inside(x, y):
            return None
        point_m = cv2.perspectiveTransform(
            np.float32([[[x, y]]]), self.matrix
        )[0][0]
        return {"x_m": round(float(point_m[0]), 2), "y_m": round(float(point_m[1]), 2)}

    def update(self, track_id, frame_no, x, y):
        if not self.inside(x, y):
            return None
        point_m = self.project(x, y)
        samples = self.history[track_id]
        samples.append((frame_no, point_m["x_m"], point_m["y_m"]))
        if len(samples) < config.MIN_SPEED_SAMPLES:
            return None

        first_frame, first_x, first_y = samples[0]
        last_frame, last_x, last_y = samples[-1]
        elapsed = (last_frame - first_frame) / self.fps
        if elapsed <= 0:
            return None
        raw_speed = float(np.hypot(last_x - first_x, last_y - first_y) / elapsed * 3.6)
        if raw_speed > config.MAX_VALID_SPEED:
            return None
        previous = self.smooth_speed.get(track_id, raw_speed)
        self.smooth_speed[track_id] = 0.7 * previous + 0.3 * raw_speed
        return self.smooth_speed[track_id]

    def direction(self, track_id):
        samples = self.history.get(track_id)
        if not samples or len(samples) < 2:
            return "UNKNOWN"
        vertical_displacement = samples[-1][2] - samples[0][2]
        if abs(vertical_displacement) < 0.5:
            return "UNKNOWN"
        return "DOWN" if vertical_displacement > 0 else "UP"
