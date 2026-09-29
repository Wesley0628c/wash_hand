"""
Data Collection Tool for Wash Hand Gestures
Records 30-frame sequence clips of hand landmarks & features for model training.
"""

import os
import time
import argparse
import numpy as np
import cv2
import pandas as pd
from typing import List

from src.camera import Camera
from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.rule_classifier import LABELS, LABEL_SHORT_ZH, LABEL_NAMES_ZH


def run_collector(
    person_id: str = "person_01",
    action: str = "inside",
    target_clips: int = 10,
    seq_length: int = 30,
    camera_id: int = 0,
    output_dir: str = "data/processed",
):
    save_dir = os.path.join(output_dir, person_id)
    os.makedirs(save_dir, exist_ok=True)
    labels_csv_path = os.path.join("data", "labels.csv")

    detector = HandDetector(max_num_hands=2)
    cam = Camera(source=camera_id)
    if not cam.open():
        print(f"[Error] Failed to open camera {camera_id}")
        return

    print(f"\n==========================================")
    print(f" Wash Hand Data Collector")
    print(f" Participant: {person_id}")
    print(f" Target Action: {action} ({LABEL_SHORT_ZH.get(action, '')})")
    print(f" Goal: Record {target_clips} clips ({seq_length} frames each)")
    print(f" Press SPACE to start recording a clip")
    print(f" Press Q to quit")
    print(f"==========================================\n")

    recorded_clips = 0
    is_recording = False
    is_counting_down = False
    countdown_start = 0.0
    recorded_frames: List[np.ndarray] = []
    prev_left = None
    prev_right = None

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Flip horizontally for selfie mirror feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        left_hand, right_hand, results = detector.process(frame)
        canvas = detector.draw_hands(frame, results)

        features = extract_hand_features(left_hand, right_hand, prev_left, prev_right)
        feat_vec = feature_dict_to_vector(features)
        prev_left = left_hand
        prev_right = right_hand

        # State Handling: Countdown
        if is_counting_down:
            elapsed = time.time() - countdown_start
            cd = 3 - int(elapsed)
            if cd > 0:
                cv2.putText(
                    canvas,
                    f"Get Ready: {cd}",
                    (w // 2 - 150, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 255, 255),
                    4,
                )
            else:
                is_counting_down = False
                is_recording = True
                recorded_frames = []

        # State Handling: Recording
        elif is_recording:
            recorded_frames.append(feat_vec)
            progress = len(recorded_frames) / seq_length
            bar_w = int(w * 0.6)
            x0 = int(w * 0.2)
            y0 = h - 60
            cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + 20), (50, 50, 50), -1)
            cv2.rectangle(
                canvas,
                (x0, y0),
                (x0 + int(bar_w * progress), y0 + 20),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                canvas,
                f"RECORDING [{len(recorded_frames)}/{seq_length}]",
                (x0, y0 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

            if len(recorded_frames) >= seq_length:
                # Save clip as .npy
                ts = int(time.time() * 1000)
                file_name = f"{action}_{ts}.npy"
                file_path = os.path.join(save_dir, file_name)
                clip_data = np.array(recorded_frames, dtype=np.float32)  # shape: (30, feature_dim)
                np.save(file_path, clip_data)

                # Record in CSV
                entry = pd.DataFrame(
                    [
                        {
                            "person_id": person_id,
                            "action": action,
                            "file_path": file_path,
                            "frames": seq_length,
                            "feature_dim": clip_data.shape[1],
                            "timestamp": ts,
                        }
                    ]
                )
                if os.path.exists(labels_csv_path):
                    entry.to_csv(labels_csv_path, mode="a", header=False, index=False)
                else:
                    os.makedirs(os.path.dirname(labels_csv_path), exist_ok=True)
                    entry.to_csv(labels_csv_path, index=False)

                recorded_clips += 1
                is_recording = False
                print(f"[*] Saved clip {recorded_clips}/{target_clips}: {file_path}")

        # Top HUD Banner
        cv2.rectangle(canvas, (0, 0), (w, 60), (20, 20, 20), -1)
        info_text = f"Action: {action} ({LABEL_SHORT_ZH.get(action, '')}) | Participant: {person_id} | Clips: {recorded_clips}/{target_clips}"
        cv2.putText(
            canvas, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.imshow("Wash Hand Data Collector", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # SPACE
            if not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start = time.time()

        if recorded_clips >= target_clips:
            print(f"\n[+] Finished collecting {target_clips} clips for action '{action}'!")
            break

    cam.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect 30-frame hand wash gesture clips")
    parser.add_argument("--person", type=str, default="person_01", help="Participant ID")
    parser.add_argument(
        "--action",
        type=str,
        default="inside",
        choices=list(LABELS.values()),
        help="Target action",
    )
    parser.add_argument("--clips", type=int, default=10, help="Number of clips to collect")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    run_collector(
        person_id=args.person,
        action=args.action,
        target_clips=args.clips,
        camera_id=args.camera,
    )
