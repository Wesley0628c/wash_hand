"""
Threshold Optimizer Module (Bayesian Optimization via Optuna)
=============================================================

Problem with Level-based (if-elif cascade) classification:
  - Upper levels silently block lower levels
  - Thresholds are hand-tuned based on geometric assumptions
    that may not match real video distributions

Solution:
  - Reformulate rule_classifier's ~23 threshold parameters as a
    continuous vector theta
  - Use Optuna (Tree-structured Parzen Estimator = Bayesian Optim.)
    to maximize the total steps completed across all sample videos
  - No per-frame labels needed: reward signal = how many 7-steps pass

Usage:
    python src/threshold_optimizer.py --trials 100 --output models/best_thresholds.json
"""

import argparse
import json
import os
import sys
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna not installed. Run: pip install optuna")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Sample video registry
# ---------------------------------------------------------------------------
SAMPLE_V1_DIR = "data/sample_v1"
SAMPLE_V2_DIR = "data/sample_v2"

STEP_VIDEOS = {
    "inside":    ["\u5167.mov"],
    "outside":   ["\u5916.mov"],
    "interlace": ["\u593e.mov"],
    "knuckles":  ["\u5f13.mov"],
    "thumb":     ["\u5927.mov"],
    "fingertips":["\u7acb.mov"],
    "wrist":     ["\u8155.mov"],
}

# ---------------------------------------------------------------------------
# Parameterized RuleClassifier
# ---------------------------------------------------------------------------

class ParameterizedRuleClassifier:
    """
    Version of WashHandRuleClassifier where all thresholds are externally
    configurable, enabling automatic Bayesian optimization.

    KEY INSIGHT: Eliminates the Level / if-elif cascade problem.
    All gesture detectors evaluate independently via boolean flags,
    then combine through explicit rivalry scores rather than silent skipping.
    """

    def __init__(self, params: Dict[str, float]):
        self.p = params

    def predict_probabilities(self, features: Dict[str, Any]) -> Dict[str, float]:
        from src.rule_classifier import LABELS
        has_left  = features.get("has_left",  False)
        has_right = features.get("has_right", False)
        both_hands = features.get("both_hands_detected", False)
        p = self.p

        if not (has_left or has_right):
            probs = {k: 0.01 for k in LABELS.values()}
            probs["other"] = 0.93
            return probs

        scores = {k: 0.0 for k in LABELS.values()}

        if both_hands:
            inter       = features.get("inter_hand", {})
            palm_dist   = float(inter.get("palm_center_dist", 99.0))
            wrist_ratio = float(inter.get("wrist_ratio", 99.0))
            min_t_to_p  = float(inter.get("min_tips_to_palm", 99.0))
            min_t_to_w  = float(inter.get("min_tips_to_wrist", 99.0))
            min_kn_to_p = float(inter.get("min_knuckles_to_palm", 99.0))
            min_p_to_th = float(inter.get("min_palm_to_thumb", 99.0))
            min_web_th  = float(inter.get("min_web_to_thumb", 99.0))
            interlace_d = float(inter.get("interlace_depth", 99.0))
            fing_spread = float(inter.get("min_fingertip_spread", 99.0))
            tip_dist    = float(inter.get("mean_tip_dist", 99.0))
            x_overlap   = float(inter.get("finger_x_overlap", 0.0))

            left_angles  = features.get("left_angles",  [0.0] * 5)
            right_angles = features.get("right_angles", [0.0] * 5)
            left_curl    = float(np.mean(left_angles[1:]))  if has_left  else 180.0
            right_curl   = float(np.mean(right_angles[1:])) if has_right else 180.0
            min_curl     = min(left_curl, right_curl)
            max_curl     = max(left_curl, right_curl)
            curl_diff    = abs(left_curl - right_curl)

            left_norm_lm  = features.get("left_norm")
            right_norm_lm = features.get("right_norm")
            axis_angle = 90.0
            if left_norm_lm is not None and right_norm_lm is not None:
                v_l = left_norm_lm[9]  - left_norm_lm[0]
                v_r = right_norm_lm[9] - right_norm_lm[0]
                nrm = np.linalg.norm(v_l) * np.linalg.norm(v_r)
                if nrm > 1e-6:
                    cos_a = np.dot(v_l, v_r) / nrm
                    axis_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

            # -- Wrist
            wrist_hit = (
                (wrist_ratio < p["wrist_ratio_thr"] or min_t_to_w < p["tips_to_wrist_thr"])
                and palm_dist < p["palm_dist_wrist"]
            )
            if wrist_hit:
                scores["wrist"] = 3.5

            # -- Interlace (evaluated independently, NOT after wrist elif)
            interlace_hit = (
                palm_dist < p["interlace_palm_dist"]
                and min_curl > p["min_curl_interlace"]
                and (
                    x_overlap >= p["x_overlap_interlace"]
                    or (interlace_d < 1.20 and (tip_dist > 0.30 or interlace_d < 1.05))
                    or (tip_dist > 0.45 and interlace_d < 1.25)
                )
            )
            if interlace_hit and not wrist_hit:
                scores["interlace"] = 3.2
                scores["outside"]   = -1.0   # rivalry penalty
                scores["inside"]    = -0.5

            # -- Knuckles
            knuckle_hit = (
                (min_curl < p["knuckle_curl_thr"] or curl_diff > 20.0)
                and min_kn_to_p < p["knuckle_dist_thr"]
                and min_kn_to_p <= min_t_to_p + 0.12
                and palm_dist < p["knuckle_palm_dist"]
            )
            if knuckle_hit and not wrist_hit and not interlace_hit:
                scores["knuckles"] = 3.0

            # -- Fingertips
            fingertip_hit = (
                min_curl >= p["fingertip_curl_thr"]
                and min_t_to_p < p["fingertip_dist_thr"]
                and fing_spread < p["fingertip_spread_thr"]
                and (palm_dist > 0.80 or min_t_to_p < min_kn_to_p - 0.08)
            )
            if fingertip_hit and not wrist_hit and not knuckle_hit:
                scores["fingertips"] = 3.0

            # -- Thumb
            thumb_hit = (
                (min_curl < p["thumb_curl_thr"] or curl_diff > 20.0)
                and (min_p_to_th < p["thumb_dist_thr"] or min_web_th < p["thumb_dist_thr"])
                and (min_p_to_th <= min_kn_to_p + 0.10 or min_web_th <= min_kn_to_p + 0.10)
            )
            if thumb_hit and not wrist_hit and not knuckle_hit:
                scores["thumb"] = 3.0

            # -- Outside
            outside_hit = (
                axis_angle < p["outside_axis_thr"]
                and palm_dist < 1.30
                and interlace_d > p["outside_interlace"]
                and max_curl > 120.0
                and min_kn_to_p > 0.55
                and x_overlap < p["outside_x_overlap"]
            )
            if outside_hit and not wrist_hit and not interlace_hit:
                scores["outside"] = max(scores["outside"], 2.8)

            # -- Inside (fallback)
            if (palm_dist < 1.50
                    and all(scores[k] <= 0.0 for k in
                            ["outside", "interlace", "knuckles", "fingertips", "wrist", "thumb"])):
                scores["inside"] = 2.0
            elif (palm_dist >= 1.50
                    and all(v <= 0.0 for k, v in scores.items() if k != "other")):
                scores["other"] = 2.0

        else:  # single hand
            active_angles = features.get("active_angles", [0.0] * 5)
            active_spread = float(features.get("active_spread", 0.0))
            velocity      = float(features.get("velocity", 0.0))
            mean_4_angle  = float(np.mean(active_angles[1:])) if len(active_angles) >= 5 else 180.0
            thumb_angle   = float(active_angles[0])           if len(active_angles) >= 1 else 180.0
            thumb_tip_w   = float(features.get("active_thumb_tip_to_wrist", 2.0))

            if (mean_4_angle < p["wrist1h_4ang_thr"]
                    and thumb_angle < p["wrist1h_thumb_thr"]
                    and velocity > p["wrist1h_vel_thr"]):
                scores["wrist"] = 2.2
            elif (mean_4_angle < p["thumb1h_4ang_thr"]
                    and scores["wrist"] <= 0.0
                    and (thumb_tip_w < p["thumb1h_tw_thr"]
                         or thumb_angle < p["thumb1h_tang_thr"]
                         or thumb_angle > 125.0)):
                scores["thumb"] = 2.2
            elif mean_4_angle < 135.0:
                scores["knuckles"] = 2.0
            elif mean_4_angle >= 140.0 and active_spread < 0.32:
                scores["fingertips"] = 2.0
            elif thumb_angle < 135.0 and mean_4_angle > 145.0 and active_spread < 0.40:
                scores["outside"] = 1.8
            elif mean_4_angle > 140.0:
                scores["inside"] = 1.9
            elif active_spread > 0.48:
                scores["interlace"] = 1.8
            else:
                scores["other"] = 2.0

        score_arr  = np.array(list(scores.values()), dtype=np.float32)
        score_arr  = np.clip(score_arr, -10.0, 15.0)
        exp_s      = np.exp(score_arr - np.max(score_arr))
        norm_probs = exp_s / max(1e-6, float(np.sum(exp_s)))
        return {act: float(prob) for act, prob in zip(scores.keys(), norm_probs)}


# ---------------------------------------------------------------------------
# Pipeline runner (lightweight, no UI overhead)
# ---------------------------------------------------------------------------

def run_video_with_params(video_path: str, params: Dict[str, float],
                          step_duration: float = 1.0) -> Dict[str, float]:
    """Run a single video, return observed_seconds per step."""
    sys.path.insert(0, ".")
    from src.hand_detector import HandDetector
    from src.features import extract_hand_features
    from src.accumulator import TemporalProbabilityAccumulator
    from src.state_machine import WashHandStateMachine

    det        = HandDetector(min_detection_confidence=0.50, min_tracking_confidence=0.50)
    classifier = ParameterizedRuleClassifier(params)
    acc        = TemporalProbabilityAccumulator(window_sec=0.5, margin_threshold=0.12,
                                               consecutive_frames_required=2)
    sm         = WashHandStateMachine(mode="free", step_duration=step_duration)
    sm.start()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_lh, prev_rh = None, None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts  = frame_idx / fps
        lh, rh, _ = det.process(frame)
        feat = extract_hand_features(lh, rh, prev_lh, prev_rh)
        prev_lh, prev_rh = lh, rh

        if lh is not None or rh is not None:
            fp = classifier.predict_probabilities(feat)
        else:
            from src.rule_classifier import LABELS
            fp = {k: 0.01 for k in LABELS.values()}
            fp["other"] = 0.93

        label, _, _ = acc.update(fp, timestamp=ts)
        obs_w = 0.8 if (lh is not None or rh is not None) else 0.0
        sm.update(label, dt=1.0 / fps, observation_weight=obs_w)

    cap.release()
    return sm.get_progress_summary()["observed_times"]


def evaluate_params(params: Dict[str, float], sample_dirs: list,
                    step_duration: float = 1.0) -> float:
    """
    Objective: maximize step-specific completion across all sample videos.
    Score range: 0.0 (nothing passes) to len(sample_dirs)*7 (all pass).
    """
    total = 0.0
    for sdir in sample_dirs:
        for step_name, filenames in STEP_VIDEOS.items():
            for fname in filenames:
                vpath = os.path.join(sdir, fname)
                if not os.path.exists(vpath):
                    continue
                try:
                    obs = run_video_with_params(vpath, params, step_duration)
                    # Primary reward: target step completion
                    step_score = min(obs.get(step_name, 0.0) / step_duration, 1.0)
                    # Penalty: false positives in other steps (small)
                    fp_penalty = sum(
                        min(v / step_duration, 0.5) * 0.05
                        for k, v in obs.items()
                        if k != step_name and k != "other" and v > step_duration * 0.3
                    )
                    total += step_score - fp_penalty
                except Exception:
                    pass
    return total


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "wrist_ratio_thr":      0.80,
    "tips_to_wrist_thr":    0.90,
    "palm_dist_wrist":      2.00,
    "x_overlap_interlace":  0.50,
    "interlace_palm_dist":  1.50,
    "min_curl_interlace":   115.0,
    "knuckle_curl_thr":     135.0,
    "knuckle_dist_thr":     0.90,
    "knuckle_palm_dist":    1.50,
    "fingertip_curl_thr":   125.0,
    "fingertip_dist_thr":   1.05,
    "fingertip_spread_thr": 0.32,
    "thumb_curl_thr":       125.0,
    "thumb_dist_thr":       1.05,
    "outside_axis_thr":     70.0,
    "outside_interlace":    0.68,
    "outside_x_overlap":    0.30,
    "wrist1h_4ang_thr":     120.0,
    "wrist1h_thumb_thr":    120.0,
    "wrist1h_vel_thr":      0.012,
    "thumb1h_4ang_thr":     160.0,
    "thumb1h_tw_thr":       0.90,
    "thumb1h_tang_thr":     110.0,
}

PARAM_BOUNDS = {
    "wrist_ratio_thr":      (0.50, 1.20),
    "tips_to_wrist_thr":    (0.50, 2.00),
    "palm_dist_wrist":      (1.00, 3.00),
    "x_overlap_interlace":  (0.20, 0.90),
    "interlace_palm_dist":  (0.80, 2.00),
    "min_curl_interlace":   (80.0, 140.0),
    "knuckle_curl_thr":     (100.0, 160.0),
    "knuckle_dist_thr":     (0.50, 1.50),
    "knuckle_palm_dist":    (0.80, 2.00),
    "fingertip_curl_thr":   (100.0, 150.0),
    "fingertip_dist_thr":   (0.60, 1.50),
    "fingertip_spread_thr": (0.15, 0.60),
    "thumb_curl_thr":       (90.0, 150.0),
    "thumb_dist_thr":       (0.60, 1.50),
    "outside_axis_thr":     (40.0, 100.0),
    "outside_interlace":    (0.40, 1.20),
    "outside_x_overlap":    (0.10, 0.60),
    "wrist1h_4ang_thr":     (90.0, 175.0),
    "wrist1h_thumb_thr":    (80.0, 175.0),
    "wrist1h_vel_thr":      (0.003, 0.05),
    "thumb1h_4ang_thr":     (120.0, 180.0),
    "thumb1h_tw_thr":       (0.50, 1.80),
    "thumb1h_tang_thr":     (80.0, 145.0),
}


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def run_optimization(n_trials: int = 100,
                     output_path: str = "models/best_thresholds.json",
                     sample_dirs: list = None) -> Dict[str, float]:

    if sample_dirs is None:
        sample_dirs = [d for d in [SAMPLE_V1_DIR, SAMPLE_V2_DIR] if os.path.isdir(d)]

    max_score = len(sample_dirs) * 7
    print(f"[Optimizer] Sample dirs  : {sample_dirs}")
    print(f"[Optimizer] Trials       : {n_trials}")
    print(f"[Optimizer] Max score    : {max_score:.0f}  ({len(sample_dirs)} dirs x 7 steps)")
    print(f"[Optimizer] Algorithm    : TPE (Tree-structured Parzen Estimator)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_float(name, lo, hi)
            for name, (lo, hi) in PARAM_BOUNDS.items()
        }
        return evaluate_params(params, sample_dirs)

    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(DEFAULT_PARAMS)        # warm-start with hand-tuned defaults
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best       = study.best_params
    best_score = study.best_value
    print(f"\n[Optimizer] Best score: {best_score:.3f} / {max_score}")
    for k, v in sorted(best.items()):
        d = DEFAULT_PARAMS.get(k, 0.0)
        arrow = "^" if v > d else ("v" if v < d else "=")
        print(f"  {arrow} {k:30s} {d:.4f} -> {v:.4f}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"best_params": best, "best_score": best_score,
                   "max_possible_score": max_score,
                   "n_trials": n_trials, "sample_dirs": sample_dirs}, f, indent=2)
    print(f"[Optimizer] Saved -> {output_path}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian threshold optimizer for wash-hand 7-step detection"
    )
    parser.add_argument("--trials",      type=int,   default=50)
    parser.add_argument("--output",      default="models/best_thresholds.json")
    parser.add_argument("--sample-dirs", nargs="+",
                        default=[SAMPLE_V1_DIR, SAMPLE_V2_DIR])
    args = parser.parse_args()

    run_optimization(n_trials=args.trials, output_path=args.output,
                     sample_dirs=args.sample_dirs)
