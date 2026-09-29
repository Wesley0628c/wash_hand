"""
Real-time Wash Hand Detection & Feedback Pipeline
Combines Camera, MediaPipe Hand Tracking, Feature Extraction,
Classifiers (Rule-based & LSTM), 1.0s Temporal Probability Accumulator, State Machine, and UI.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from collections import deque
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

from src.camera import Camera
from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.rule_classifier import WashHandRuleClassifier, LABELS, NAME_TO_LABEL, FEEDBACK_ZH
from src.ml_classifier import WashHandMLClassifier
from src.state_machine import WashHandStateMachine
from src.accumulator import TemporalProbabilityAccumulator
from src.ui import WashHandHUD


class RealtimeWashHandDetector:
    """Main Real-time WebCam Application Engine."""

    def __init__(
        self,
        mode: str = "hybrid",
        model_path: str = "models/wash_hand_xgb.joblib",
        guide_mode: str = "free",  # "free" (arbitrary random order) or "sequence" (strict sequential order)
        step_duration: float = 2.0,
        window_sec: float = 0.5,
        source: Any = 0,
        flip_camera: bool = True,
    ):
        self.mode = mode
        self.model_path = model_path
        self.flip_camera = flip_camera
        self.camera = Camera(source=source)
        self.detector = HandDetector(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Classifiers
        self.ml_classifier = WashHandMLClassifier(model_path=model_path, hybrid_with_rules=(mode == "hybrid"))
        self.rule_classifier = WashHandRuleClassifier()

        self.accumulator = TemporalProbabilityAccumulator(window_sec=window_sec, margin_threshold=0.15)
        self.state_machine = WashHandStateMachine(mode=guide_mode, step_duration=step_duration)
        self.hud = WashHandHUD()

        self.prev_left = None
        self.prev_right = None
        self.prev_timestamp = time.time()

    def run(self):
        if not self.camera.open():
            print(f"[Error] 無法開啟攝影機 (Source: {self.camera.source})，請確認 WebCam 連接或權限。")
            return

        print("\n=======================================================")
        print(" 🧼 七步洗手即時辨識系統 (WebCam 即時模式已啟動)")
        print(f" 當前模型: {self.mode.upper()} | 決策窗口: {self.accumulator.window_sec}s")
        print(" 鍵盤快捷鍵：")
        print("   [Q] 離開程式 (Quit)")
        print("   [R] 重置洗手進度 (Reset)")
        print("   [M] 切換模式 (教學順序 Sequence / 自由 Free)")
        print("   [C] 切換分類器 (Hybrid ↔ ML ↔ Rule-based)")
        print("   [F] 水平翻轉攝影機畫面 (Flip Mirror)")
        print("=======================================================\n")

        self.state_machine.start()

        while True:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                break

            now = time.time()
            dt = max(0.001, min(0.1, now - self.prev_timestamp))
            self.prev_timestamp = now

            # Flip for mirror selfie perspective (if enabled)
            if self.flip_camera:
                frame = cv2.flip(frame, 1)

            # 1. MediaPipe Hand Detection with CLAHE & Temporal Tracking
            left_hand, right_hand, results = self.detector.process(frame)
            canvas = self.detector.draw_hands(frame.copy(), results)

            # 2. Feature Extraction
            features = extract_hand_features(
                left_hand, right_hand, self.prev_left, self.prev_right
            )
            self.prev_left = left_hand
            self.prev_right = right_hand

            # 3. Action Probability Calculation
            if self.mode == "hybrid":
                self.ml_classifier.hybrid_with_rules = True
                frame_probs = self.ml_classifier.predict_probabilities(features)
                mode_display = f"{self.state_machine.mode.capitalize()} | HYBRID (XGB+Rule)"
            elif self.mode == "ml":
                self.ml_classifier.hybrid_with_rules = False
                frame_probs = self.ml_classifier.predict_probabilities(features)
                mode_display = f"{self.state_machine.mode.capitalize()} | ML (XGBoost)"
            else:
                frame_probs = self.rule_classifier.predict_probabilities(features)
                mode_display = f"{self.state_machine.mode.capitalize()} | RULE-BASED"

            # 4. Temporal Winner-Take-All Probability Accumulation
            decided_label, decided_conf, _ = self.accumulator.update(frame_probs, timestamp=now)
            feedback_msg = FEEDBACK_ZH.get(decided_label, "請依七步口訣持續搓洗")

            # 5. State Machine Update (separate fresh observation from held ghost frames)
            meta = self.detector.last_metadata
            is_fresh_observation = meta.get("is_observed", False)
            just_completed, completed_step = self.state_machine.update(
                decided_label, dt=dt, is_observed=is_fresh_observation
            )
            if just_completed and completed_step:
                print(f"[🎉] 恭喜！已完成步驟: {completed_step}")

            # 6. UI HUD Rendering
            hands_status = {
                "left": meta.get("left_status") == "observed",
                "right": meta.get("right_status") == "observed",
            }
            progress = self.state_machine.get_progress_summary()
            final_frame = self.hud.draw_hud(
                canvas,
                detected_label=decided_label,
                confidence=decided_conf,
                feedback_msg=feedback_msg,
                progress_summary=progress,
                fps=self.camera.fps,
                mode_str=mode_display,
                hands_status=hands_status,
            )

            cv2.imshow("Wash Hand Detect — Real-time Action Recognition", final_frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.state_machine.reset()
                self.accumulator.reset()
                self.ml_classifier.reset()
                print("[*] 洗手會話已重置 (Session Reset)。")
            elif key == ord("m"):
                new_mode = "free" if self.state_machine.mode == "sequence" else "sequence"
                self.state_machine.mode = new_mode
                self.state_machine.reset()
                self.accumulator.reset()
                print(f"[*] 切換洗手導引模式為: {new_mode}")
            elif key == ord("c"):
                if self.mode == "hybrid":
                    self.mode = "ml"
                elif self.mode == "ml":
                    self.mode = "rule"
                else:
                    self.mode = "hybrid"
                self.accumulator.reset()
                self.ml_classifier.reset()
                print(f"[*] 切換分類器為: {self.mode.upper()}")
            elif key == ord("f"):
                self.flip_camera = not self.flip_camera
                print(f"[*] 鏡像翻轉: {'開啟' if self.flip_camera else '關閉'}")

        self.camera.release()
        self.detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Wash Hand Detection Application via WebCam")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "ml", "rule"], help="Classifier mode")
    parser.add_argument("--model-path", type=str, default="models/wash_hand_xgb.joblib", help="Path to trained XGBoost model")
    parser.add_argument("--guide-mode", type=str, default="free", choices=["free", "sequence"], help="Guide flow mode (free: random order, sequence: strict order)")
    parser.add_argument("--step-duration", type=float, default=2.0, help="Seconds required per step")
    parser.add_argument("--window-sec", type=float, default=0.5, help="Sliding window seconds for probability integration")
    parser.add_argument("--camera", default=0, help="WebCam index (0, 1, 2) or video file path")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror horizontal flip")
    args = parser.parse_args()

    # If camera is digit string, convert to int
    source_val = int(args.camera) if str(args.camera).isdigit() else args.camera

    app = RealtimeWashHandDetector(
        mode=args.mode,
        model_path=args.model_path,
        guide_mode=args.guide_mode,
        step_duration=args.step_duration,
        window_sec=args.window_sec,
        source=source_val,
        flip_camera=not args.no_flip,
    )
    app.run()
