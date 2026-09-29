"""
Ground Truth Segment-Level Evaluation Module
Evaluates Wash Hand Detect system against ground-truth time segments,
computing Frame Accuracy, Segment Precision, Recall, F1-score, and Confusion Matrix.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Dict, List, Tuple, Any, Optional
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.hand_detector import HandDetector
from src.features import extract_hand_features
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH, FEEDBACK_ZH
from src.ml_classifier import WashHandMLClassifier
from src.accumulator import TemporalProbabilityAccumulator
from src.state_machine import WashHandStateMachine
from src.ui import WashHandHUD

# Ground Truth Segment Definitions
BENCHMARK_GROUND_TRUTH = {
    "wash_7steps_yt.mp4": [
        (0.0, 4.3, "other"),
        (4.3, 8.3, "inside"),
        (8.3, 13.5, "outside"),
        (13.5, 19.0, "interlace"),
        (19.0, 24.4, "knuckles"),
        (24.4, 30.7, "thumb"),
        (30.7, 34.4, "fingertips"),
        (34.4, 39.2, "wrist"),
        (39.2, 50.8, "other"),
    ],
    "wash_7steps_yt.mov": [
        (0.0, 4.3, "other"),
        (4.3, 8.3, "inside"),
        (8.3, 13.5, "outside"),
        (13.5, 19.0, "interlace"),
        (19.0, 24.4, "knuckles"),
        (24.4, 30.7, "thumb"),
        (30.7, 34.4, "fingertips"),
        (34.4, 39.2, "wrist"),
        (39.2, 50.8, "other"),
    ],
    "wash_7steps_yt2.mp4": [
        (0.0, 19.0, "other"),
        (19.0, 23.0, "inside"),
        (24.0, 27.0, "outside"),
        (28.0, 31.0, "interlace"),
        (32.0, 36.0, "knuckles"),
        (37.0, 41.0, "thumb"),
        (42.0, 46.0, "fingertips"),
        (47.0, 51.0, "wrist"),
        (52.0, 93.0, "other"),
    ],
    "wash_7steps_yt2.mov": [
        (0.0, 19.0, "other"),
        (19.0, 23.0, "inside"),
        (24.0, 27.0, "outside"),
        (28.0, 31.0, "interlace"),
        (32.0, 36.0, "knuckles"),
        (37.0, 41.0, "thumb"),
        (42.0, 46.0, "fingertips"),
        (47.0, 51.0, "wrist"),
        (52.0, 93.0, "other"),
    ],
    "內.mov": [(0.0, 999.0, "inside")],
    "外.mov": [(0.0, 999.0, "outside")],
    "夾.mov": [(0.0, 999.0, "interlace")],
    "弓.mov": [(0.0, 999.0, "knuckles")],
    "大.mov": [(0.0, 999.0, "thumb")],
    "立.mov": [(0.0, 999.0, "fingertips")],
    "腕.mov": [(0.0, 999.0, "wrist")],
}


def get_ground_truth_label(time_sec: float, segments: List[Tuple[float, float, str]]) -> str:
    for start_t, end_t, label in segments:
        if start_t <= time_sec < end_t:
            return label
    return "other"


def evaluate_video(
    video_path: str,
    output_annotated_path: Optional[str] = None,
    show_window: bool = False,
    model_type: str = "hybrid",
    window_sec: float = 0.5,
    margin_threshold: float = 0.15,
) -> Dict[str, Any]:
    video_name = os.path.basename(video_path)
    if video_name not in BENCHMARK_GROUND_TRUTH:
        print(f"[!] 警告: 未知影片名稱 '{video_name}'，無預設 Ground Truth 標註。不計算監督式指標 (未標註影片不假定為全 other)。")
        segments = None
    else:
        segments = BENCHMARK_GROUND_TRUTH[video_name]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = HandDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if model_type == "ml":
        classifier = WashHandMLClassifier(hybrid_with_rules=False)
        mode_str = "ML (XGBoost)"
    elif model_type == "hybrid":
        classifier = WashHandMLClassifier(hybrid_with_rules=True, rule_weight=0.3)
        mode_str = "Hybrid (XGBoost + Rules)"
    else:
        classifier = WashHandRuleClassifier()
        mode_str = "Rule-based V2"

    accumulator = TemporalProbabilityAccumulator(window_sec=window_sec, margin_threshold=margin_threshold)
    state_machine = WashHandStateMachine(mode="free", step_duration=1.0)
    state_machine.start()
    hud = WashHandHUD()

    writer = None
    if output_annotated_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_annotated_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

    y_true = []
    y_pred = []
    prev_left = None
    prev_right = None

    frame_idx = 0
    print(f"\n=======================================================")
    print(f"📊 正在評估影片: {video_path}")
    print(f"   解析度: {width}x{height} | 總幀數: {total_frames} | FPS: {fps:.1f}")
    print(f"   模型類型: {mode_str} | 決策窗口: {window_sec}s")
    if output_annotated_path:
        print(f"   輸出帶 HUD 影片: {output_annotated_path}")
    print(f"=======================================================")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        dt = 1.0 / fps
        gt_label = get_ground_truth_label(timestamp, segments) if segments is not None else "unlabeled"

        left_hand, right_hand, results = detector.process(frame)
        features = extract_hand_features(
            left_hand, right_hand, prev_left_hand=prev_left, prev_right_hand=prev_right
        )
        probs = classifier.predict_probabilities(features)
        pred_label, pred_conf, _ = accumulator.update(probs, timestamp=timestamp)

        is_obs = detector.last_metadata.get("is_observed", False)
        state_machine.update(pred_label, dt=dt, is_observed=is_obs)
        summary = state_machine.get_progress_summary()
        feedback_msg = FEEDBACK_ZH.get(pred_label, "請依七步口訣持續搓洗")

        prev_left = left_hand
        prev_right = right_hand

        y_true.append(gt_label)
        y_pred.append(pred_label)
        frame_idx += 1

        # Draw HUD and annotations if saving or displaying
        if writer or show_window:
            canvas = detector.draw_hands(frame.copy(), results)
            annotated_frame = hud.draw_hud(
                frame=canvas,
                detected_label=pred_label,
                confidence=pred_conf,
                feedback_msg=f"[GT: {LABEL_SHORT_ZH.get(gt_label, gt_label)}] {feedback_msg}",
                progress_summary=summary,
                fps=fps,
                mode_str=mode_str,
            )
            if writer:
                writer.write(annotated_frame)
            if show_window:
                cv2.imshow("Wash Hand Detect - Evaluation", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if writer:
        writer.release()
        print(f"[✅] 已成功儲存視覺化辨識結果影片至: {output_annotated_path}")
    if show_window:
        cv2.destroyAllWindows()
    detector.close()

    # Calculate metrics if ground truth is available
    if segments is None:
        print("\n[!] 該影片未登記 Ground Truth 標註，不計算監督式指標 (Precision / Recall / Confusion Matrix)。")
        from collections import Counter
        pred_counts = Counter(y_pred)
        print("📊 【預測動作分佈】")
        for p_label, cnt in pred_counts.most_common():
            zh_name = LABEL_SHORT_ZH.get(p_label, p_label)
            print(f"   - {zh_name:2s} ({p_label:10s}): {cnt:5d} 幀 ({(cnt/len(y_pred))*100:5.1f}%)")
        return {"report": None, "segment_accuracies": {}, "confusion_matrix": None}

    target_names = [LABELS[i] for i in range(8)]
    report = classification_report(y_true, y_pred, labels=target_names, zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=target_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=target_names)

    # Segment Accuracy calculation (for active washing segments)
    active_segments = [s for s in segments if s[2] != "other"]
    segment_accuracies = {}
    for s_start, s_end, s_label in active_segments:
        start_idx = int(s_start * fps)
        end_idx = min(len(y_true), int(s_end * fps))
        if end_idx > start_idx:
            sub_true = y_true[start_idx:end_idx]
            sub_pred = y_pred[start_idx:end_idx]
            correct = sum(1 for t, p in zip(sub_true, sub_pred) if p == s_label)
            acc = correct / len(sub_true)
            segment_accuracies[s_label] = acc

    print("\n📈 【分類準確率與 F1-Score 報告】")
    print(report_text)

    print("\n🎯 【七步各動作時間區間辨識率 (Segment Recall)】")
    for act, acc in segment_accuracies.items():
        zh_name = LABEL_SHORT_ZH.get(act, act)
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"   - {zh_name:2s} ({act:10s}): [{bar}] {acc*100:5.1f}%")

    print("\n🧩 【混淆矩陣 (Confusion Matrix)】")
    header = "       " + " ".join([f"{LABEL_SHORT_ZH[l]:>4s}" for l in target_names])
    print(header)
    for idx, row in enumerate(cm):
        row_str = " ".join([f"{val:4d}" for val in row])
        print(f"{LABEL_SHORT_ZH[target_names[idx]]:>4s}: {row_str}")

    return {
        "report": report,
        "segment_accuracies": segment_accuracies,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 7-step hand wash detector on ground truth video.")
    parser.add_argument("--video", type=str, default="data/raw/wash_7steps_yt.mp4", help="Video path")
    parser.add_argument("--output", type=str, default=None, help="Output annotated video path (.mp4)")
    parser.add_argument("--show", action="store_true", help="Show real-time window")
    parser.add_argument("--model-type", type=str, default="hybrid", choices=["rule", "ml", "hybrid"], help="Classifier type")
    parser.add_argument("--window-sec", type=float, default=0.5, help="Temporal window size in seconds")
    parser.add_argument("--margin", type=float, default=0.15, help="Hysteresis margin")
    args = parser.parse_args()

    evaluate_video(
        args.video,
        output_annotated_path=args.output,
        show_window=args.show,
        model_type=args.model_type,
        window_sec=args.window_sec,
        margin_threshold=args.margin,
    )
