"""
Video Clip Extraction Utility Module
Extracts 7 distinct action video clips from raw benchmark washing videos
for training, fine-tuning, and offline dataset augmentation.
"""

import os
import cv2
from typing import List, Tuple

VIDEO_SEGMENTS = {
    "wash_7steps_yt.mp4": {
        "output_dir": "data/clips/video1_yt",
        "segments": [
            ("01_inside", 4.3, 8.3),
            ("02_outside", 8.3, 13.5),
            ("03_interlace", 13.5, 19.0),
            ("04_knuckles", 19.0, 24.4),
            ("05_thumb", 24.4, 30.7),
            ("06_fingertips", 30.7, 34.4),
            ("07_wrist", 34.4, 39.2),
        ],
    },
    "wash_7steps_yt2.mp4": {
        "output_dir": "data/clips/video2_yt2",
        "segments": [
            ("01_inside", 19.5, 21.5),
            ("02_outside", 24.5, 26.5),
            ("03_interlace", 28.5, 30.5),
            ("04_knuckles", 33.0, 35.0),
            ("05_thumb", 38.0, 40.0),
            ("06_fingertips", 43.0, 45.0),
            ("07_wrist", 48.0, 50.0),
        ],
    },
}


def cut_video_clips(raw_dir: str = "data/raw"):
    for video_name, config in VIDEO_SEGMENTS.items():
        src_path = os.path.join(raw_dir, video_name)
        if not os.path.exists(src_path):
            print(f"[!] 找不到來源影片: {src_path}")
            continue

        out_dir = config["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(src_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n=======================================================")
        print(f" 🎬 開始剪輯來源影片: {video_name}")
        print(f" 解析度: {width}x{height} | FPS: {fps:.2f} | 總長: {total_frames/fps:.1f}s")
        print(f" 輸出資料夾: {out_dir}")
        print(f"=======================================================")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for label_name, start_sec, end_sec in config["segments"]:
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            out_file = os.path.join(out_dir, f"{label_name}.mp4")

            writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            curr_frame = start_frame
            saved_count = 0
            while curr_frame <= end_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                writer.write(frame)
                saved_count += 1
                curr_frame += 1

            writer.release()
            duration_sec = saved_count / fps
            print(f"  [✓] 剪輯完成: {label_name}.mp4 ({start_sec}s -> {end_sec}s, 共 {saved_count} 幀, {duration_sec:.1f}s)")

        cap.release()

    print("\n🎉 全部 7 步動作片段剪輯完成！")


if __name__ == "__main__":
    cut_video_clips()
