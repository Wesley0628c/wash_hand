"""
Camera & Video Source Helper Module
Provides flexible frame capture from webcam, video files, or synthetic test streams.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, Union


class Camera:
    """Wrapper around cv2.VideoCapture with automatic FPS calculation and resource handling."""

    def __init__(self, source: Union[int, str] = 0, width: int = 1280, height: int = 720):
        self.source = source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_time = 0.0
        self.fps = 0.0
        self._is_synthetic = False

    def open(self) -> bool:
        if isinstance(self.source, str) and self.source == "synthetic":
            self._is_synthetic = True
            return True

        if isinstance(self.source, str) and not str(self.source).isdigit():
            # Video file path
            self.cap = cv2.VideoCapture(self.source)
            return self.cap.isOpened()

        # WebCam index probing (with AVFoundation on macOS)
        import platform
        is_macos = (platform.system() == "Darwin")
        backend = cv2.CAP_AVFOUNDATION if is_macos else cv2.CAP_ANY

        preferred_idx = int(self.source)
        # Probe preferred index first, then fallback to other common indices (0, 1, 2)
        indices_to_try = [preferred_idx] + [i for i in [0, 1, 2] if i != preferred_idx]

        for idx in indices_to_try:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                cap = cv2.VideoCapture(idx)

            if cap.isOpened():
                # Test read a frame and verify it is not all black/zeros
                ret, frame = False, None
                for _ in range(3):  # warm-up capture
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        break

                if ret and frame is not None and np.mean(frame) > 0.5:
                    self.cap = cap
                    self.source = idx
                    if is_macos:
                        print(f"[*] 成功啟用 macOS 鏡頭 (Device Index: {idx}, AVFoundation, 解析度: {frame.shape[1]}x{frame.shape[0]})")
                    else:
                        print(f"[*] 成功啟用 WebCam 鏡頭 (Device Index: {idx})")
                    return True
                else:
                    # Device opened but returned blank/black frame
                    cap.release()

        # Fallback: try opening preferred index directly with default backend
        self.cap = cv2.VideoCapture(preferred_idx)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True

        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._is_synthetic:
            # Generate a clean dark background test canvas
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:] = (30, 30, 30)
            self._update_fps()
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            # If video file reached end, can loop if needed
            if isinstance(self.source, str) and self.source != "synthetic":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret:
                return False, None

        self._update_fps()
        return True, frame

    def _update_fps(self):
        curr_time = time.time()
        if self.prev_time > 0:
            dt = curr_time - self.prev_time
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else (1.0 / dt)
        self.prev_time = curr_time

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
