"""
Offline / Batch Video Evaluation Script
Runs the Wash Hand Detection Pipeline on a video file with 1.0s Windowed Maximum-Probability Integration.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import cv2
import numpy as np
from collections import Counter

from src.pipeline import WashHandPipeline, PipelineConfig
from src.rule_classifier import LABEL_NAMES_ZH, LABEL_SHORT_ZH, LABELS, FEEDBACK_ZH
from src.ui import WashHandHUD


def evaluate_video(
    video_path: str,
    output_annotated_path: str = None,
    model_type: str = "rule",
    guide_mode: str = "free",
    step_duration: float = 1.0,
    window_sec: float = 0.5,
    max_frames: int = 1500,
    roi_mode: str = "none",
    ori_scale: float = 1.0,
    show_window: bool = False,
    trace_path: str = None,
    verbose: bool = False,
):
    if not os.path.exists(video_path):
        print(f"[!] 找不到影片檔案: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] 無法開啟影片: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ori_scale = max(0.2, float(ori_scale))
    width = int(orig_width * ori_scale)
    height = int(orig_height * ori_scale)

    use_right_roi = (roi_mode == "right") or (roi_mode == "auto" and width > height * 1.3)

    print(f"\n=======================================================")
    print(f"🎬 測試影片: {video_path}")
    print(f"   解析度: {orig_width}x{orig_height}" + (f" -> 放大為 {width}x{height} (ori: {ori_scale:.2f}x)" if ori_scale != 1.0 else "") + f" | FPS: {fps:.1f} | 總幀數: {total_frames}")
    print(f"   模型類型: {model_type.upper()} | 決策窗口: {window_sec} 秒")
    print(f"   ROI 模式: {roi_mode} ({'啟用右側子母畫面裁切' if use_right_roi else '全畫面模式 (Full-Frame)'})")
    print(f"=======================================================")

    pipeline_cfg = PipelineConfig(
        model_type=model_type,
        model_path="models/wash_hand_xgb.joblib",
        crop_split_screen=use_right_roi,
        window_sec=window_sec,
        margin_threshold=0.12,
        consecutive_frames_required=2,
        guide_mode=guide_mode,
        step_duration=step_duration,
    )
    pipeline = WashHandPipeline(pipeline_cfg)

    writer = None
    if output_annotated_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_annotated_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_annotated_path, fourcc, fps, (width, height))

    trace = None
    if trace_path:
        os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
        trace = open(trace_path, "w", encoding="utf-8")

    predictions = Counter()
    observed_predictions = Counter()
    hand_count_stats = Counter()  # 0, 1, 2 observed hands
    ghost_frames_count = 0

    start_time = time.time()
    frame_idx = 0
    hud = WashHandHUD()

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames > 0 and frame_idx >= max_frames):
            break

        if ori_scale != 1.0:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        frame_idx += 1
        dt = 1.0 / fps
        simulated_time = frame_idx * dt

        # Process through standardized pipeline
        res, mediapipe_results = pipeline.process_frame(
            frame, timestamp_sec=simulated_time, crop_split_screen=use_right_roi
        )

        if trace:
            trace.write(json.dumps({
                "frame": frame_idx, "time": simulated_time,
                "hands": res.hand_status, "raw": res.raw_label,
                "scores": res.raw_scores, "integrated": res.integrated_probs,
                "display": res.display_label, "credit": res.credit_label,
                "weight": res.quality_score, "reason": res.evidence_reason,
                "relative_motion_speed": res.features.get("relative_motion_speed", 0.0),
                "shape_speed": res.features.get("shape_speed", 0.0),
                "inter_hand": res.features.get("inter_hand", {}),
                "back_evidence": res.features.get("back_evidence", {}),
                "completed": res.just_completed_step,
            }, ensure_ascii=False) + "\n")

        # Track hand observation statistics
        n_obs = res.num_hands_observed
        hand_count_stats[n_obs] += 1
        if res.hand_status.get("left") == "held" or res.hand_status.get("right") == "held":
            ghost_frames_count += 1

        predictions[res.display_label] += 1
        observed_predictions[res.observed_label] += 1

        # Visualization — pass roi_context to draw_hands for correct landmark projection
        roi_ctx = pipeline.detector._last_roi_context  # corrects ROI landmark to full-frame coords
        if use_right_roi:
            xmin = int(width * 0.42)
            canvas = frame.copy()
            cv2.rectangle(canvas, (xmin, 0), (width, height), (0, 255, 0), 2)
            cv2.putText(canvas, "[ROI Area]", (xmin + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            annotated = pipeline.detector.draw_hands(canvas, mediapipe_results, roi_context=roi_ctx)
        else:
            annotated = pipeline.detector.draw_hands(frame.copy(), mediapipe_results, roi_context=None)


        hands_status = {
            "left": res.hand_status.get("left") == "observed",
            "right": res.hand_status.get("right") == "observed",
        }
        progress_summary = pipeline.state_machine.get_progress_summary()
        annotated = hud.draw_hud(
            frame=annotated,
            detected_label=res.display_label,
            confidence=res.confidence,
            feedback_msg=res.feedback_msg,
            fps=fps,
            mode_str=f"{model_type.upper()} | Win: {window_sec}s",
            hands_status=hands_status,
            progress_summary=progress_summary,
        )

        if writer:
            writer.write(annotated)

        if show_window:
            cv2.imshow("Video Test - Wash Hand Detect", annotated)
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break

        if verbose and (frame_idx % 15 == 0 or res.just_completed_step):
            t_sec = frame_idx / fps
            l_stat = "觀測" if hands_status["left"] else ("保留" if res.hand_status.get("left") == "held" else "無")
            r_stat = "觀測" if hands_status["right"] else ("保留" if res.hand_status.get("right") == "held" else "無")
            act_zh = LABEL_SHORT_ZH.get(res.display_label, res.display_label)
            obs_zh = LABEL_SHORT_ZH.get(res.observed_label, res.observed_label)
            completed_str = f" 🌟 步驟【{LABEL_SHORT_ZH.get(res.just_completed_step, res.just_completed_step)}】達標！" if res.just_completed_step else ""
            print(f"⏱️ [{t_sec:5.1f}s | 幀 {frame_idx:4d}] 顯示: 【{act_zh:2s}】 (信心: {res.confidence*100:4.1f}%) | 有效: 【{obs_zh:2s}】 | 手部: [左:{l_stat} 右:{r_stat}]{completed_str}")
        elif not verbose and (frame_idx % 60 == 0 or frame_idx == total_frames):
            target_max = min(total_frames, max_frames) if max_frames > 0 else total_frames
            print(f"   進度: {frame_idx}/{target_max} 幀 ({(frame_idx/target_max)*100:.1f}%)")

    cap.release()
    pipeline.detector.close()
    if trace:
        trace.close()
    if writer:
        writer.release()
        print(f"[+] 已儲存標註影片至: {output_annotated_path}")

    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    summary = pipeline.state_machine.get_progress_summary()

    # Detailed Hand Detection & Tracking Analysis
    obs_0 = hand_count_stats[0]
    obs_1 = hand_count_stats[1]
    obs_2 = hand_count_stats[2]
    at_least_one = obs_1 + obs_2

    print(f"\n📊 【評估成果統計】")
    print(f"   總處理幀數: {frame_idx} 幀 (耗時: {elapsed:.2f}s, 平均速度: {frame_idx/max(0.001, elapsed):.1f} FPS)")
    print(f"   ── 手部真實觀測分佈 ──")
    print(f"   - 0 隻手 (未檢出): {obs_0} 幀 ({(obs_0/max(1,frame_idx))*100:.1f}%)")
    print(f"   - 1 隻手 (單手觀測): {obs_1} 幀 ({(obs_1/max(1,frame_idx))*100:.1f}%)")
    print(f"   - 2 隻手 (雙手同時): {obs_2} 幀 ({(obs_2/max(1,frame_idx))*100:.1f}%)")
    print(f"   - 手部偵測率 (至少單手): {at_least_one}/{frame_idx} ({(at_least_one/max(1,frame_idx))*100:.1f}%)")
    print(f"   - 真正雙手偵測率 (雙手同時): {obs_2}/{frame_idx} ({(obs_2/max(1,frame_idx))*100:.1f}%)")
    print(f"   - 歷史狀態保留 (Ghost Held): {ghost_frames_count} 幀 ({(ghost_frames_count/max(1,frame_idx))*100:.1f}%)")

    print(f"\n🖐️ 【時序平滑後的動作分布 (顯示標籤)】")
    for act, cnt in predictions.most_common():
        zh = LABEL_NAMES_ZH.get(act, act)
        print(f"   - {zh:8s} ({act:12s}): {cnt:5d} 幀 ({(cnt/max(1,frame_idx))*100:.1f}%)")

    print(f"\n⏱️ 【可靠真實觀測下的動作分布 (排除無當前證據)】")
    for act, cnt in observed_predictions.most_common():
        zh = LABEL_NAMES_ZH.get(act, act)
        print(f"   - {zh:8s} ({act:12s}): {cnt:5d} 幀 ({(cnt/max(1,frame_idx))*100:.1f}%)")

    print(f"\n🏆 【七步洗手完成狀態】")
    for step_name in ["inside", "outside", "interlace", "knuckles", "thumb", "fingertips", "wrist"]:
        obs_time = summary.get("observed_times", {}).get(step_name, 0.0)
        total_time = summary["step_times"].get(step_name, 0.0)
        is_done = step_name in summary["completed_steps"]
        status = f"✅ 已達標 (可靠觀測: {obs_time:.1f}s / 總計: {total_time:.1f}s)" if is_done else f"⏳ 累積可靠時間: {obs_time:.1f}s / {step_duration}s"
        zh = LABEL_NAMES_ZH.get(step_name, step_name)
        print(f"   - {zh}: {status}")

    print(f"   完成步驟數: {summary['completed_count']}/{summary['total_steps']}")
    print(f"   總體狀態: {'🎉 全部步驟完成！' if summary['is_completed'] else '進行中'}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video with Wash Hand Detector Pipeline")
    parser.add_argument("--video", "--input", dest="video", type=str, default="data/annotated_eval_yt.mp4", help="Path to input video")
    parser.add_argument("--output", "--output-video", dest="output", type=str, default=None, help="Optional output annotated video path")
    parser.add_argument("--model-type", type=str, default="rule", choices=["hybrid", "ml", "rule"])
    parser.add_argument("--guide-mode", type=str, default="free", choices=["free", "sequence"])
    parser.add_argument("--step-duration", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument("--roi", type=str, default="none", choices=["none", "right", "auto"], help="ROI cropping strategy (default: none / full-frame)")
    parser.add_argument("--ori", nargs="?", const=1.5, type=float, default=1.0, help="放大原始影像顯示倍率 (Scale factor for original video display, e.g. 1.2, 1.5, 2.0; 預設 1.0，帶 --ori 未填數字預設 1.5)")
    parser.add_argument("--max-frames", type=int, default=1500)
    parser.add_argument("--trace", default=None, help="Write frame evidence as JSONL")
    parser.add_argument("--show", action="store_true", help="Display visual playback window while testing")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed real-time detection progress per second")
    args = parser.parse_args()

    evaluate_video(
        video_path=args.video,
        output_annotated_path=args.output,
        model_type=args.model_type,
        guide_mode=args.guide_mode,
        step_duration=args.step_duration,
        window_sec=args.window_sec,
        max_frames=args.max_frames,
        roi_mode=args.roi,
        ori_scale=args.ori,
        show_window=args.show,
        trace_path=args.trace,
        verbose=args.verbose,
    )
