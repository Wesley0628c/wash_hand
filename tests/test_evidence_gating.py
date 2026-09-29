"""Regression cases for false completion, shared coordinates and motion timing."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.features import extract_hand_features
from src.pipeline import WashHandPipeline, PipelineConfig
from src.state_machine import WashHandStateMachine
from src.rule_classifier import WashHandRuleClassifier, LABELS
from src.accumulator import TemporalProbabilityAccumulator


def hand(offset=0.0):
    points = np.zeros((21, 3), np.float32)
    points[:, 0] = np.linspace(-0.3, 0.3, 21)
    points[:, 1] = np.linspace(0.1, 1.0, 21)
    points[0] = [0, 0, 0]
    points[5] = [-0.3, 0.5, 0]
    points[17] = [0.3, 0.5, 0]
    return points + [offset, 0, 0]


def test_overlap_preserves_separation_and_rotation():
    left, right = hand(), hand(10)
    assert extract_hand_features(left, right)['inter_hand']['finger_x_overlap'] == 0
    near = hand(0.1)
    expected = extract_hand_features(left, near)['inter_hand']['finger_x_overlap']
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    actual = extract_hand_features(left @ rotation.T, near @ rotation.T)['inter_hand']['finger_x_overlap']
    assert actual == expected


def test_relative_motion_is_fps_invariant_and_rejects_common_translation():
    left, right = hand(), hand(0.2)
    values = []
    for fps in (15, 30, 60):
        current = left + [0.5 / fps, 0, 0]
        values.append(extract_hand_features(current, right, left, right, dt=1/fps)['relative_motion_speed'])
    np.testing.assert_allclose(values, values[0], rtol=1e-5)
    f = extract_hand_features(left + 0.1, right + 0.1, left, right)
    assert f['relative_motion_speed'] < 1e-5


def test_progress_does_not_bias_classification():
    sm = WashHandStateMachine(mode='free')
    before = sm.get_state_prior()
    sm.update('inside', 1.0)
    assert sm.get_state_prior() == before


def test_scattered_evidence_does_not_complete_step():
    sm = WashHandStateMachine(mode='free', step_duration=1)
    for _ in range(8):
        sm.update('inside', 0.2)
        sm.update('other', 2.0, observation_weight=0)
    assert sm.observed_times['inside'] > 1
    assert 'inside' not in sm.completed_steps


def test_sequence_does_not_complete_at_85_percent():
    sm = WashHandStateMachine(mode='sequence', step_duration=1)
    sm.update('inside', 0.9)
    sm.update('outside', 0.1, observation_weight=0)
    assert not sm.completed_steps
    sm.update('outside', 0.1)
    assert not sm.completed_steps


def fake_pipeline(monkeypatch):
    import src.pipeline as module
    class Detector:
        def __init__(self, **kwargs):
            self.last_metadata = {'left_status': 'observed', 'right_status': 'observed', 'raw_num_hands': 2}
            self.left, self.right = hand(), hand(0.2)
        def process(self, *args, **kwargs):
            return self.left, self.right, None
    monkeypatch.setattr(module, 'HandDetector', Detector)
    pipeline = WashHandPipeline(PipelineConfig())
    pipeline.rule_classifier.predict_probabilities = lambda features: {k: (0.93 if k == 'inside' else 0.01) for k in LABELS.values()}
    return pipeline


def test_static_dual_hands_never_receive_credit(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    for i in range(100):
        result, _ = pipeline.process_frame(None, i / 30)
    assert result.display_label == 'inside'
    assert not pipeline.state_machine.completed_steps
    assert sum(pipeline.state_machine.observed_times.values()) == 0


def test_held_hand_never_creates_dual_hand_evidence(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    pipeline.detector.last_metadata['right_status'] = 'held'
    for i in range(60):
        pipeline.detector.left = hand(np.sin(i * 0.2) * 0.1)
        result, _ = pipeline.process_frame(None, i / 30)
    assert not result.features['both_hands_observed']
    assert result.features['contains_held']
    assert result.credit_label == 'other'
    assert not pipeline.state_machine.completed_steps


def test_supported_moving_hands_can_complete(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    for i in range(120):
        pipeline.detector.left = hand(i * 0.01)
        result, _ = pipeline.process_frame(None, i / 30)
    assert 'inside' in pipeline.state_machine.completed_steps


def test_transition_display_does_not_credit_previous_step(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    for i in range(30):
        pipeline.detector.left = hand(i * 0.01)
        pipeline.process_frame(None, i / 30)
    before = pipeline.state_machine.observed_times['inside']
    pipeline.rule_classifier.predict_probabilities = lambda features: {k: (0.93 if k == 'outside' else 0.01) for k in LABELS.values()}
    pipeline.detector.left = hand(0.3)
    result, _ = pipeline.process_frame(None, 1.0)
    assert result.display_label == 'inside'
    assert result.credit_label == 'other'
    assert pipeline.state_machine.observed_times['inside'] == before


def test_single_hand_classifier_can_recognize_candidate():
    classifier = WashHandRuleClassifier()
    features = {"has_left": True, "has_right": False, "both_hands_detected": False,
                "active_angles": [150, 170, 170, 170, 170], "active_spread": 0.3}
    assert classifier.predict(features)[0] == "inside"


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_hand_can_complete_without_prior_dual_observation(monkeypatch, side):
    pipeline = fake_pipeline(monkeypatch)
    missing = "right" if side == "left" else "left"
    setattr(pipeline.detector, missing, None)
    pipeline.detector.last_metadata = {f"{side}_status": "observed", f"{missing}_status": "missing", "raw_num_hands": 1}
    for i in range(120):
        points = hand()
        points[8, 0] += i * 0.1
        setattr(pipeline.detector, side, points)
        result, _ = pipeline.process_frame(None, i / 30)
    assert result.num_hands_observed == 1
    assert result.credit_label == "inside"
    assert "inside" in pipeline.state_machine.completed_steps


def test_static_single_hand_does_not_receive_credit(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    pipeline.detector.right = None
    pipeline.detector.last_metadata['right_status'] = 'missing'
    for i in range(120):
        result, _ = pipeline.process_frame(None, i / 30)
    assert result.credit_label == "other"
    assert not pipeline.state_machine.completed_steps


def test_large_tracking_jump_cannot_receive_credit(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    for i in range(30):
        pipeline.detector.left = hand(i * 0.01)
        pipeline.process_frame(None, i / 30)
    pipeline.detector.left = hand(10)
    result, _ = pipeline.process_frame(None, 1.0)
    assert result.credit_label == 'other'
    assert result.evidence_reason == 'static_or_tracking_jump'


def test_long_frame_gap_resets_unfinished_evidence(monkeypatch):
    pipeline = fake_pipeline(monkeypatch)
    for i in range(20):
        pipeline.detector.left = hand(i * 0.01)
        pipeline.process_frame(None, i / 30)
    assert pipeline.state_machine.contiguous_times['inside'] > 0
    result, _ = pipeline.process_frame(None, 5.0)
    assert result.credit_label == 'other'
    assert pipeline.state_machine.contiguous_times['inside'] == 0


def test_strong_spike_still_requires_elapsed_confirmation():
    acc = TemporalProbabilityAccumulator(window_sec=0.001, confirmation_sec=0.06, consecutive_frames_required=2)
    probs = {k: (0.93 if k == 'inside' else 0.01) for k in LABELS.values()}
    assert acc.update(probs, 0)[0] == 'other'
    assert acc.update(probs, 0.03)[0] == 'other'
    assert acc.update(probs, 0.07)[0] == 'inside'
    spike = {k: (0.93 if k == 'outside' else 0.01) for k in LABELS.values()}
    assert acc.update(spike, 0.08)[0] == 'inside'
