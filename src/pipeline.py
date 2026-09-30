"""
Unified Wash Hand Detection Pipeline Module
Provides a single, standardized pipeline (WashHandPipeline) for real-time video,
offline batch evaluation, and benchmark analysis.
"""

import os
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH, FEEDBACK_ZH
from src.ml_classifier import WashHandMLClassifier
from src.accumulator import TemporalProbabilityAccumulator
from src.state_machine import WashHandStateMachine
from src.back_contact import BackContactTracker


@dataclass
class PipelineConfig:
    """Centralized configuration for wash hand detection pipeline."""
    model_type: str = "rule"  # "hybrid", "rule", "ml" — default is pure rule-based
    model_path: str = "models/wash_hand_xgb.joblib"
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50
    ghost_frames_threshold: int = 4
    crop_split_screen: bool = False
    window_sec: float = 0.50
    margin_threshold: float = 0.12
    consecutive_frames_required: int = 2
    rule_weight: float = 0.25
    step_duration: float = 1.0
    guide_mode: str = "free"  # "free" or "sequence"
    credit_min_confidence: float = 0.45
    credit_min_margin: float = 0.12
    evidence_duration: float = 0.20
    min_relative_motion_speed: float = 0.02  # palm lengths / second
    max_relative_motion_speed: float = 30.0  # reject real tracking teleport jumps
    min_shape_speed: float = 0.02  # normalized shape change / second
    max_shape_speed: float = 30.0
    max_frame_gap: float = 0.20
    outside_min_score: float = 0.60
    outside_role_seconds: float = 0.40
    outside_entry_seconds: float = 0.50
    outside_occlusion_seconds: float = 0.15



@dataclass
class FrameResult:
    """Structured per-frame output across the detection and classification lifecycle."""
    timestamp_sec: float
    frame_id: int
    left_hand: Optional[np.ndarray]
    right_hand: Optional[np.ndarray]
    num_hands_observed: int
    hand_status: Dict[str, str]
    is_observed: bool
    quality_score: float
    features: Dict[str, Any]
    feature_vector: np.ndarray
    raw_scores: Dict[str, float]
    raw_label: str
    integrated_probs: Dict[str, float]
    display_label: str
    observed_label: str
    confidence: float
    feedback_msg: str
    credit_label: str = "other"
    evidence_reason: str = ""
    just_completed_step: Optional[str] = None
    is_session_completed: bool = False


class WashHandPipeline:
    """
    Unified end-to-end pipeline encapsulating hand detection, coordinate normalization,
    feature extraction, hierarchical rule & ML classification, temporal smoothing,
    and wash progress state tracking.
    """

    def __init__(self, config: Optional[PipelineConfig] = None, detector=None):
        self.config = config or PipelineConfig()

        self.detector = detector if detector is not None else HandDetector(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            ghost_frames_threshold=self.config.ghost_frames_threshold,
            crop_split_screen=self.config.crop_split_screen,
        )

        self.rule_classifier = WashHandRuleClassifier()

        self.ml_classifier = None
        if self.config.model_type in ("hybrid", "ml"):
            self.ml_classifier = WashHandMLClassifier(
                model_path=self.config.model_path,
                hybrid_with_rules=(self.config.model_type == "hybrid"),
                rule_weight=self.config.rule_weight,
            )

        consecutive_req = self.config.consecutive_frames_required
        self.accumulator = TemporalProbabilityAccumulator(
            window_sec=self.config.window_sec,
            margin_threshold=self.config.margin_threshold,
            consecutive_frames_required=consecutive_req,
        )

        self.state_machine = WashHandStateMachine(
            mode=self.config.guide_mode,
            step_duration=self.config.step_duration,
        )
        self.state_machine.start()

        self.prev_left: Optional[np.ndarray] = None
        self.prev_right: Optional[np.ndarray] = None
        self.frame_idx: int = 0
        self._prev_timestamp: Optional[float] = None
        self._evidence_history = deque()
        self._previous_both_observed = False
        self.back_tracker = BackContactTracker(
            self.config.outside_min_score, self.config.outside_role_seconds,
            self.config.outside_entry_seconds, self.config.outside_occlusion_seconds,
        )

    def reset(self):
        """Reset internal pipeline history."""
        self.detector.reset()
        self.accumulator.reset()
        self.state_machine.reset()
        if self.ml_classifier is not None:
            self.ml_classifier.reset()
        self.prev_left = None
        self.prev_right = None
        self.frame_idx = 0
        self._prev_timestamp = None
        self._evidence_history = deque()
        self._previous_both_observed = False
        self.back_tracker.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: Optional[float] = None,
        crop_split_screen: Optional[bool] = None,
    ) -> Tuple[FrameResult, Any]:
        """
        Process a single BGR video frame through the complete pipeline.
        Returns:
            (FrameResult, mediapipe_results)
        """
        self.frame_idx += 1
        ts = timestamp_sec if timestamp_sec is not None else float(self.frame_idx / 30.0)

        # Compute real dt from consecutive timestamps (avoids hardcoded 1/30)
        if self._prev_timestamp is not None:
            dt = max(1e-4, ts - self._prev_timestamp)
        else:
            dt = 1.0 / 30.0
        self._prev_timestamp = ts

        # 1. Detection
        left_hand, right_hand, results = self.detector.process(
            frame, crop_split_screen=crop_split_screen
        )
        meta = self.detector.last_metadata

        # Held coordinates provide temporary tracking continuity during foam/occlusion (up to ghost_frames_threshold)
        left_status = meta.get("left_status", "missing")
        right_status = meta.get("right_status", "missing")

        # Active hands for feature extraction:
        # If at least one hand is currently observed, allow using detector hands (including
        # temporal held coordinates) so that dual-hand interaction geometry does not collapse in 1 frame.
        if left_status == "observed" or right_status == "observed":
            active_left = left_hand
            active_right = right_hand
        else:
            active_left = None
            active_right = None

        # 2. Features
        features = extract_hand_features(
            active_left, active_right, self.prev_left, self.prev_right, dt=dt
        )
        features["both_hands_observed"] = left_status == right_status == "observed"
        features["contains_held"] = "held" in (left_status, right_status)
        features["observed_sides"] = [side for side, status in (("left", left_status), ("right", right_status))
                                      if status == "observed"]
        self.back_tracker.update(features, ts, dt, features["both_hands_observed"],
                                 self._previous_both_observed)
        self._previous_both_observed = features["both_hands_observed"]
        vec_160 = feature_dict_to_vector(features)

        # 3. Model Classification
        if active_left is not None or active_right is not None:
            if self.config.model_type in ("hybrid", "ml") and self.ml_classifier is not None:
                frame_probs = self.ml_classifier.predict_probabilities(features)
            else:
                # rule-only: correct API call (no timestamp param)
                frame_probs = self.rule_classifier.predict_probabilities(features)
        else:
            # No hands: reset ML temporal buffer to prevent stale feature carry-over
            if self.ml_classifier is not None:
                self.ml_classifier.reset()
            frame_probs = {k: 0.01 for k in LABELS.values()}
            frame_probs["other"] = 0.93

        raw_label = max(frame_probs, key=frame_probs.get)

        # 4. Temporal Smoothing Accumulator
        label, conf, integrated = self.accumulator.update(frame_probs, timestamp=ts)

        # 5. Separate display smoothing from current evidence eligible for credit.
        has_observation = (left_status == "observed" or right_status == "observed")
        has_continuity = ((active_left is not None and self.prev_left is not None)
                          or (active_right is not None and self.prev_right is not None))
        both_active = (active_left is not None and active_right is not None)

        ranked = sorted(frame_probs.values(), reverse=True)
        if features["both_hands_observed"] and self.prev_left is not None and self.prev_right is not None:
            motion_speed = float(features.get("relative_motion_speed", 0.0))
            min_speed, max_speed = self.config.min_relative_motion_speed, self.config.max_relative_motion_speed
        else:
            motion_speed = float(features.get("shape_speed", 0.0))
            min_speed, max_speed = self.config.min_shape_speed, self.config.max_shape_speed

        if not has_observation:
            reason = "no_current_observation"
        elif not has_continuity or dt > self.config.max_frame_gap:
            reason = "tracking_reacquired"
        elif label == "other" or raw_label != label:
            reason = "uncertain_or_transition"
        elif ranked[0] < self.config.credit_min_confidence or ranked[0] - ranked[1] < self.config.credit_min_margin:
            reason = "ambiguous_scores"
        elif not min_speed <= motion_speed <= max_speed:
            reason = "static_or_tracking_jump"
        elif label == "outside" and not features["back_evidence"]["credit_ready"]:
            reason = "outside_contact_unconfirmed"
        else:
            reason = "supported"

        self._evidence_history.append((ts, label if reason == "supported" else "other", min(dt, self.config.max_frame_gap)))
        while self._evidence_history and self._evidence_history[0][0] < ts - 0.5:
            self._evidence_history.popleft()
        support_seconds = sum(duration for _, action, duration in self._evidence_history if action == label)
        stable = support_seconds >= self.config.evidence_duration
        obs_weight = 1.0 if reason == "supported" and stable else 0.0
        if reason == "supported" and not stable:
            reason = "confirming"
        credit_label = label if obs_weight else "other"
        is_reliable_observation = obs_weight > 0
        just_completed, completed_name = self.state_machine.update(
            label, dt=dt, observation_weight=obs_weight
        )

        feedback_msg = self.state_machine.get_fsm_feedback(label, confidence=conf)

        if not obs_weight and not self.state_machine.is_completed:
            feedback_msg = "動作尚待確認／手部遮擋，暫停計時"
        self.prev_left = active_left
        self.prev_right = active_right

        result = FrameResult(
            timestamp_sec=ts,
            frame_id=self.frame_idx,
            left_hand=left_hand,
            right_hand=right_hand,
            num_hands_observed=meta.get("raw_num_hands", 0),
            hand_status={"left": left_status, "right": right_status},
            is_observed=is_reliable_observation,
            quality_score=obs_weight,
            features=features,
            feature_vector=vec_160,
            raw_scores=frame_probs,
            raw_label=raw_label,
            integrated_probs=integrated,
            display_label=label,
            observed_label=label if is_reliable_observation else "other",
            confidence=conf,
            feedback_msg=feedback_msg,
            credit_label=credit_label,
            evidence_reason=reason,
            just_completed_step=completed_name,
            is_session_completed=self.state_machine.is_completed,
        )

        return result, results
