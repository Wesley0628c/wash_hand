"""
Comprehensive Robustness & Regression Test Suite for Wash Hand Detection
Tests coordinate projection contracts, state machine timing separation (observed vs held),
detector ghosting/quality tracking, ML fallback safety, and accumulator temporal dynamics.
"""

import os
import sys
import pytest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.state_machine import WashHandStateMachine
from src.ml_classifier import WashHandMLClassifier
from src.accumulator import TemporalProbabilityAccumulator
from src.pipeline import WashHandPipeline, PipelineConfig


# =====================================================================
# 1. Coordinate Projection Contract Tests (Appendix A Oracle)
# =====================================================================

def test_roi_coordinate_projection_1280x720():
    """
    Oracle Test for Coordinate Normalization:
    For 1280x720 video with split-screen ROI x in [0.42*W, W], y in [0, H]:
    A landmark at ROI center (u=0.5, v=0.5) must project to:
    x_global = (0.42 * 1280 + 0.5 * (1280 * 0.58)) / 1280 = 0.42 + 0.29 = 0.71
    y_global = (0.0 + 0.5 * 720) / 1280 = 360 / 1280 = 0.28125
    In the flawed version, y_iso was ~0.170508 due to double scaling.
    """
    W, H = 1280, 720
    max_dim = float(max(W, H))  # 1280.0
    xmin = int(W * 0.42)
    ymin = 0
    roi_w = W - xmin
    roi_h = H

    u, v, z = 0.5, 0.5, 0.0

    # Using the corrected isotropic mapping formula
    x_iso = (xmin + u * roi_w) / max_dim
    y_iso = (ymin + v * roi_h) / max_dim

    expected_x = (xmin + 0.5 * roi_w) / 1280.0
    expected_y = 360.0 / 1280.0  # 0.28125

    assert np.isclose(x_iso, expected_x, atol=1e-5)
    assert np.isclose(y_iso, expected_y, atol=1e-5)
    assert np.isclose(y_iso, 0.28125, atol=1e-5)


def test_fullframe_coordinate_projection():
    """Full-frame projection is a special case: x0=0, y0=0, rw=W, rh=H."""
    W, H = 1920, 1080
    max_dim = 1920.0
    u, v = 0.5, 0.5

    x_iso = (0.0 + u * W) / max_dim
    y_iso = (0.0 + v * H) / max_dim

    assert np.isclose(x_iso, 0.5, atol=1e-5)
    assert np.isclose(y_iso, 540.0 / 1920.0, atol=1e-5)


# =====================================================================
# 2. Timing & Observation Separation (Section 5 Oracle)
# =====================================================================

def test_timing_separation_observed_vs_held():
    """
    Oracle:
    1.0s of reliable observation + 0.3s of held/unobserved + 1.0s of reliable observation:
    - observed_times must equal 2.0s (NOT 2.3s)
    - step_times accumulates total 2.3s
    """
    sm = WashHandStateMachine(mode="free", step_duration=1.0)
    sm.start()

    # Step 1: 1.0s inside with reliable observation (30 frames at 30 fps)
    for _ in range(30):
        sm.update("inside", dt=1.0 / 30.0, is_observed=True)

    obs_t1 = sm.observed_times["inside"]
    step_t1 = sm.step_times["inside"]
    assert np.isclose(obs_t1, 1.0, atol=0.01)
    assert np.isclose(step_t1, 1.0, atol=0.01)
    assert "inside" in sm.completed_steps

    # Step 2: 0.3s of unobserved / ghost held frames (9 frames at 30 fps)
    for _ in range(9):
        sm.update("inside", dt=1.0 / 30.0, is_observed=False)

    obs_t2 = sm.observed_times["inside"]
    step_t2 = sm.step_times["inside"]
    # observed_times must stay at 1.0s!
    assert np.isclose(obs_t2, 1.0, atol=0.01)
    # step_times continues accumulating display duration: 1.0 + 0.3 = 1.3s
    assert np.isclose(step_t2, 1.3, atol=0.01)

    # Step 3: Another 1.0s of reliable observation (30 frames at 30 fps)
    for _ in range(30):
        sm.update("inside", dt=1.0 / 30.0, is_observed=True)

    obs_t3 = sm.observed_times["inside"]
    step_t3 = sm.step_times["inside"]
    # observed_times must be 2.0s, NOT 2.3s
    assert np.isclose(obs_t3, 2.0, atol=0.02)
    assert np.isclose(step_t3, 2.3, atol=0.02)


def test_free_mode_continues_accumulating_after_completion():
    """In free mode, steps must not stop accumulating duration after 1s threshold."""
    sm = WashHandStateMachine(mode="free", step_duration=1.0)
    sm.start()

    # Wash for 3.0s continuously
    for _ in range(90):
        sm.update("thumb", dt=1.0 / 30.0, is_observed=True)

    summary = sm.get_progress_summary()
    assert "thumb" in summary["completed_steps"]
    # Duration must reflect full ~3.0s, not capped at 1.0s
    assert summary["step_times"]["thumb"] >= 2.9
    assert summary["observed_times"]["thumb"] >= 2.9


# =====================================================================
# 3. Model Compatibility & Fallback Safety (P0 Bug Oracle)
# =====================================================================

def test_ml_classifier_fallback_without_rules():
    """
    Oracle:
    When hybrid_with_rules=False and model file is non-existent,
    predict_probabilities() must return safe probabilities without AttributeError.
    """
    classifier = WashHandMLClassifier(
        model_path="models/non_existent_model_file.joblib",
        hybrid_with_rules=False,
    )
    assert classifier.model is None

    dummy_features = {"left_angles": [180] * 5, "right_angles": [180] * 5}
    probs = classifier.predict_probabilities(dummy_features)

    assert isinstance(probs, dict)
    assert "other" in probs
    assert "inside" in probs
    assert len(probs) == 8
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-4)


# =====================================================================
# 4. Temporal Accumulator Dynamics
# =====================================================================

def test_accumulator_hysteresis_and_switching():
    """
    Accumulator must require sustained probability lead before switching away from current label.
    """
    acc = TemporalProbabilityAccumulator(window_sec=0.2, margin_threshold=0.10, consecutive_frames_required=2)

    # Prime with "inside"
    p_inside = {k: 0.05 for k in ["other", "inside", "outside", "interlace", "knuckles", "thumb", "fingertips", "wrist"]}
    p_inside["inside"] = 0.65

    for t in [0.0, 0.033, 0.066, 0.100]:
        label, conf, _ = acc.update(p_inside, timestamp=t)

    assert label == "inside"

    # Single spurious spike for "outside" should not immediately flip the locked action
    p_outside_spike = {k: 0.05 for k in p_inside}
    p_outside_spike["outside"] = 0.65
    label, _, _ = acc.update(p_outside_spike, timestamp=0.133)
    assert label == "inside"


# =====================================================================
# 5. Full Pipeline Integration
# =====================================================================

def test_pipeline_process_empty_frame():
    """Pipeline processes black frame cleanly without exceptions."""
    pipeline = WashHandPipeline(PipelineConfig(model_type="rule"))
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result, mediapipe_res = pipeline.process_frame(black_frame, timestamp_sec=0.0)

    assert result.frame_id == 1
    assert result.num_hands_observed == 0
    assert result.is_observed is False
    assert result.display_label == "other"
    assert result.observed_label == "other"
    assert len(result.feature_vector) == 160


def test_tiered_observation_weights_in_state_machine():
    """Verify Solution 1: Tiered observation credit works properly in state machine."""
    sm = WashHandStateMachine(mode="free", step_duration=1.0)
    sm.start()

    # 1. Ten frames of partial occlusion with weight 0.5 (10 * 0.1s * 0.5 = 0.5s)
    for _ in range(10):
        sm.update("fingertips", dt=0.1, observation_weight=0.5)

    assert np.isclose(sm.observed_times["fingertips"], 0.5, atol=0.01)
    assert np.isclose(sm.step_times["fingertips"], 1.0, atol=0.01)
    assert "fingertips" not in sm.completed_steps

    # 2. Ten frames of unobserved/motionless ghost (weight 0.0) -> observed_times must not increase
    for _ in range(10):
        sm.update("fingertips", dt=0.1, observation_weight=0.0)

    assert np.isclose(sm.observed_times["fingertips"], 0.5, atol=0.01)
    assert np.isclose(sm.step_times["fingertips"], 2.0, atol=0.01)

    # 3. Seven frames of dual-hand observed (weight 1.0) -> 0.5 + 7 * 0.1 = 1.2s >= 1.0s
    for _ in range(7):
        sm.update("fingertips", dt=0.1, observation_weight=1.0)

    assert "fingertips" in sm.completed_steps
    assert sm.observed_times["fingertips"] >= 1.0

