"""
MediaPipe Hand Detector Wrapper Module
Extracts 21 3D landmarks for both Left and Right hands with deterministic ordering.

Key design decisions (v3.3):
- ROI mode is STRICT: when both primary+CLAHE fail inside the ROI, returns no-hand
  (never falls back to full-frame, to avoid left-side infographic interference).
- draw_hands() now accepts an optional roi_context dict to correctly project ROI-space
  MediaPipe coordinates back to the original frame for overlay drawing.
- ML feature buffer reset is called in ml_classifier.py on no-hand frames; this module
  only handles detection/tracking.
"""

from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
import mediapipe as mp


def _enhance_contrast_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply adaptive histogram equalization on L-channel to highlight finger contours under soap foam."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


class HandDetector:
    """Detects and separates Left and Right hands using MediaPipe with temporal tracking & dropout compensation."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.50,
        min_tracking_confidence: float = 0.50,
        ghost_frames_threshold: int = 4,
        crop_split_screen: bool = False,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.crop_split_screen = crop_split_screen

        # Primary tracker for continuous stream (stateful, uses tracking between frames)
        self.primary_hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Dedicated ROI tracker: stateful but restricted to ROI region
        # Using static_image_mode=False to benefit from temporal tracking within ROI
        self.roi_hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        # Stateless rescue detector for CLAHE fallback (no tracking state)
        self.rescue_hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = self.primary_hands  # Backward compatibility alias

        # Temporal tracking memory
        self.prev_left_hand: Optional[np.ndarray] = None
        self.prev_right_hand: Optional[np.ndarray] = None
        self.left_lost_count: int = 0
        self.right_lost_count: int = 0
        self.ghost_frames_threshold = ghost_frames_threshold

        # ROI context for draw_hands coordinate correction
        self._last_roi_context: Optional[Dict[str, Any]] = None

        # Observation status metadata for latest frame
        self.last_metadata: Dict[str, Any] = {
            "left_status": "missing",
            "right_status": "missing",
            "is_observed": False,
            "raw_num_hands": 0,
            "quality_score": 0.0,
            "detection_mode": "none",   # "roi_primary", "roi_clahe", "fullframe", "none"
        }

    def reset_tracking(self):
        """Reset temporal tracking history."""
        self.prev_left_hand = None
        self.prev_right_hand = None
        self.left_lost_count = 0
        self.right_lost_count = 0
        self._last_roi_context = None
        self.last_metadata = {
            "left_status": "missing",
            "right_status": "missing",
            "is_observed": False,
            "raw_num_hands": 0,
            "quality_score": 0.0,
            "detection_mode": "none",
        }

    def reset(self):
        """Alias for reset_tracking."""
        self.reset_tracking()

    def process(
        self, frame: np.ndarray, crop_split_screen: Optional[bool] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Any]:
        """
        Process a BGR OpenCV frame.

        ROI Mode (crop_split_screen=True and wide aspect ratio):
          1. Crop right 58% of frame (human washing area), rescale 2x.
          2. Run roi_hands (stateful tracker) on the ROI.
          3. If no hands found, try CLAHE enhancement with rescue_hands (stateless).
          4. STRICT: if both fail, return no-hand (do NOT fall back to full frame).

        Full-Frame Mode:
          1. Run primary_hands on full frame.
          2. If <2 hands found, try CLAHE with rescue_hands.

        Returns:
            left_hand: np.ndarray of shape (21, 3) or None  [isotropic-normalized coords]
            right_hand: np.ndarray of shape (21, 3) or None
            results: raw MediaPipe results object (for draw_hands)
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        use_split_crop = self.crop_split_screen if crop_split_screen is None else crop_split_screen
        is_roi = False
        detection_mode = "none"
        xmin, xmax = 0, w
        ymin, ymax = 0, h
        roi_w, roi_h = w, h
        scale_factor = 1

        if use_split_crop and w > h * 1.3:
            # Strict ROI mode: right 58% of frame, 5%-95% vertical
            xmin = int(w * 0.42)
            xmax = w
            ymin = int(h * 0.05)
            ymax = int(h * 0.95)
            roi_w = xmax - xmin
            roi_h = ymax - ymin
            scale_factor = 2

            roi = rgb_frame[ymin:ymax, xmin:xmax]
            roi_scaled = cv2.resize(roi, (roi_w * scale_factor, roi_h * scale_factor),
                                    interpolation=cv2.INTER_LINEAR)

            # Try stateful ROI tracker first
            results = self.roi_hands.process(roi_scaled)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                is_roi = True
                detection_mode = "roi_primary"
            else:
                # CLAHE rescue inside ROI (stateless)
                roi_clahe = _enhance_contrast_clahe(roi_scaled)
                results_clahe = self.rescue_hands.process(roi_clahe)
                if results_clahe.multi_hand_landmarks and len(results_clahe.multi_hand_landmarks) > 0:
                    results = results_clahe
                    is_roi = True
                    detection_mode = "roi_clahe"
                else:
                    # STRICT: do NOT fall back to full frame — return empty results
                    results = results_clahe  # empty results
                    detection_mode = "none"
        else:
            # Full-frame mode with stateful primary tracker
            results = self.primary_hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                detection_mode = "fullframe"
            # If <2 hands found, try CLAHE enhancement
            if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
                enhanced = _enhance_contrast_clahe(rgb_frame)
                results_clahe = self.rescue_hands.process(enhanced)
                n_orig = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                n_clahe = len(results_clahe.multi_hand_landmarks) if results_clahe.multi_hand_landmarks else 0
                if n_clahe > n_orig:
                    results = results_clahe
                    detection_mode = "fullframe_clahe"

        # Store ROI context for draw_hands coordinate correction
        self._last_roi_context = {
            "is_roi": is_roi,
            "xmin": xmin, "ymin": ymin,
            "roi_w": roi_w, "roi_h": roi_h,
            "scale_factor": scale_factor,
            "frame_w": w, "frame_h": h,
        } if is_roi else None

        # --- Extract & Normalize Landmark Coordinates ---
        curr_left_hand = None
        curr_right_hand = None
        extracted_coords = []
        extracted_labels = []

        if results.multi_hand_landmarks and results.multi_handedness:
            max_dim = float(max(w, h))
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                coords = np.zeros((21, 3), dtype=np.float32)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    if is_roi:
                        # Map from scaled-ROI space back to original frame space, then normalize
                        # lm.x, lm.y are in [0,1] relative to roi_scaled (roi_w*scale_factor)
                        x_roi_px = lm.x * roi_w * scale_factor  # px in scaled ROI
                        y_roi_px = lm.y * roi_h * scale_factor
                        # Convert to original frame pixel coordinates
                        x_orig = xmin + x_roi_px / scale_factor
                        y_orig = ymin + y_roi_px / scale_factor
                        coords[idx, 0] = x_orig / max_dim
                        coords[idx, 1] = y_orig / max_dim
                        coords[idx, 2] = lm.z * (roi_w / max_dim)
                    else:
                        coords[idx, 0] = lm.x * (w / max_dim)
                        coords[idx, 1] = lm.y * (h / max_dim)
                        coords[idx, 2] = lm.z * (w / max_dim)

                extracted_coords.append(coords)
                extracted_labels.append(label)

            # Hand ID Assignment & Temporal Tracking
            if len(extracted_coords) == 1:
                hand_cand = extracted_coords[0]
                label_cand = extracted_labels[0]

                if self.prev_left_hand is not None and self.prev_right_hand is not None:
                    d_to_left = np.linalg.norm(hand_cand[0] - self.prev_left_hand[0])
                    d_to_right = np.linalg.norm(hand_cand[0] - self.prev_right_hand[0])
                    if d_to_left < d_to_right and d_to_left < 0.25:
                        curr_left_hand = hand_cand
                    elif d_to_right < d_to_left and d_to_right < 0.25:
                        curr_right_hand = hand_cand
                    else:
                        if label_cand == "Left":
                            curr_left_hand = hand_cand
                        else:
                            curr_right_hand = hand_cand
                else:
                    if label_cand == "Left":
                        curr_left_hand = hand_cand
                    else:
                        curr_right_hand = hand_cand

            elif len(extracted_coords) >= 2:
                h1, h2 = extracted_coords[0], extracted_coords[1]
                l1, l2 = extracted_labels[0], extracted_labels[1]

                if l1 != l2 and (self.prev_left_hand is None or self.prev_right_hand is None):
                    curr_left_hand = h1 if l1 == "Left" else h2
                    curr_right_hand = h2 if l1 == "Left" else h1
                elif self.prev_left_hand is not None and self.prev_right_hand is not None:
                    cost_standard = (
                        np.linalg.norm(h1[0] - self.prev_left_hand[0])
                        + np.linalg.norm(h2[0] - self.prev_right_hand[0])
                    )
                    cost_swapped = (
                        np.linalg.norm(h2[0] - self.prev_left_hand[0])
                        + np.linalg.norm(h1[0] - self.prev_right_hand[0])
                    )
                    if cost_standard <= cost_swapped:
                        curr_left_hand, curr_right_hand = h1, h2
                    else:
                        curr_left_hand, curr_right_hand = h2, h1
                else:
                    if h1[0, 0] <= h2[0, 0]:
                        curr_left_hand, curr_right_hand = h1, h2
                    else:
                        curr_left_hand, curr_right_hand = h2, h1

        # 3. Temporal Dropout Smoothing / Ghost Tracking for Foam & Occlusion
        left_status = "missing"
        if curr_left_hand is not None:
            self.prev_left_hand = curr_left_hand.copy()
            self.left_lost_count = 0
            final_left = curr_left_hand
            left_status = "observed"
        elif self.prev_left_hand is not None and self.left_lost_count < self.ghost_frames_threshold:
            self.left_lost_count += 1
            final_left = self.prev_left_hand
            left_status = "held"
        else:
            final_left = None
            self.prev_left_hand = None
            self.left_lost_count = 0

        right_status = "missing"
        if curr_right_hand is not None:
            self.prev_right_hand = curr_right_hand.copy()
            self.right_lost_count = 0
            final_right = curr_right_hand
            right_status = "observed"
        elif self.prev_right_hand is not None and self.right_lost_count < self.ghost_frames_threshold:
            self.right_lost_count += 1
            final_right = self.prev_right_hand
            right_status = "held"
        else:
            final_right = None
            self.prev_right_hand = None
            self.right_lost_count = 0

        # Quality scoring
        if left_status == "observed" and right_status == "observed":
            quality = 1.0
        elif (left_status == "observed" and right_status == "held") or (right_status == "observed" and left_status == "held"):
            quality = 0.7
        elif left_status == "held" and right_status == "held":
            quality = 0.4
        elif left_status == "observed" or right_status == "observed":
            quality = 0.5
        elif left_status == "held" or right_status == "held":
            quality = 0.3
        else:
            quality = 0.0

        n_raw = len(extracted_coords) if (results.multi_hand_landmarks and results.multi_handedness) else 0
        self.last_metadata = {
            "left_status": left_status,
            "right_status": right_status,
            "is_observed": (left_status == "observed" and right_status == "observed"),
            "raw_num_hands": n_raw,
            "quality_score": quality,
            "detection_mode": detection_mode,
        }

        return final_left, final_right, results

    def draw_hands(
        self, frame: np.ndarray, results: Any, draw_styled: bool = True,
        roi_context: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Draw hand skeleton overlay onto frame.

        When roi_context is provided (or self._last_roi_context is set), landmarks are
        in scaled-ROI space and must be projected back to original frame coordinates.
        """
        if not results or not results.multi_hand_landmarks:
            return frame

        ctx = roi_context if roi_context is not None else self._last_roi_context

        canvas = frame.copy()
        h_frame, w_frame = canvas.shape[:2]

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            label = handedness.classification[0].label

            if ctx and ctx.get("is_roi"):
                # Project ROI-normalized coords back to original frame pixel coords for drawing
                xmin_r = ctx["xmin"]
                ymin_r = ctx["ymin"]
                roi_w_r = ctx["roi_w"]
                roi_h_r = ctx["roi_h"]
                sf = ctx["scale_factor"]

                # Build a corrected landmark list for drawing
                import mediapipe as mp_local
                from mediapipe.framework.formats import landmark_pb2
                corrected_proto = landmark_pb2.NormalizedLandmarkList()
                for lm in hand_landmarks.landmark:
                    x_orig = xmin_r + lm.x * roi_w_r * sf / sf  # simplifies to xmin + lm.x * roi_w
                    y_orig = ymin_r + lm.y * roi_h_r * sf / sf
                    corrected_proto.landmark.add(
                        x=x_orig / w_frame,
                        y=y_orig / h_frame,
                        z=lm.z,
                    )
                if draw_styled:
                    self.mp_draw.draw_landmarks(
                        canvas,
                        corrected_proto,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
                else:
                    color = (0, 255, 128) if label == "Left" else (255, 128, 0)
                    self.mp_draw.draw_landmarks(
                        canvas,
                        corrected_proto,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(220, 220, 220), thickness=2),
                    )
            else:
                if draw_styled:
                    self.mp_draw.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )
                else:
                    color = (0, 255, 128) if label == "Left" else (255, 128, 0)
                    self.mp_draw.draw_landmarks(
                        canvas,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(220, 220, 220), thickness=2),
                    )

        return canvas

    def close(self):
        if hasattr(self, "primary_hands") and self.primary_hands:
            self.primary_hands.close()
        if hasattr(self, "roi_hands") and self.roi_hands:
            self.roi_hands.close()
        if hasattr(self, "rescue_hands") and self.rescue_hands:
            self.rescue_hands.close()
