"""
Wash Hand Dataset Collector & Web Downloader Utility
Allows downloading public wash-hand training videos from YouTube/web
and automatically feeding them into the feature extraction and ML training pipeline.
"""

import os
import sys
import argparse
from typing import Optional, List, Dict
import yt_dlp

# Recommended Public 7-step Wash Hand Demonstration Video Sources
RECOMMENDED_SOURCES = [
    {
        "name": "who_standard_demo",
        "url": "https://www.youtube.com/watch?v=bSyLegFBge4",
        "desc": "WHO Standard 7-Step Hand Washing Demonstration (Sink View)",
    },
    {
        "name": "cdc_hygiene_guide",
        "url": "https://www.youtube.com/watch?v=3PmVJQUCm4E",
        "desc": "CDC Clean Hands Save Lives Hygiene Demonstration",
    },
]


def download_youtube_video(url: str, output_dir: str = "data/raw", filename: Optional[str] = None) -> str:
    """Download a video from YouTube in best mp4 format."""
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{filename}.%(ext)s" if filename else "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": out_template,
        "quiet": False,
        "no_warnings": True,
    }

    print(f"\n[*] 正在下載洗手訓練影片: {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        saved_path = ydl.prepare_filename(info)

    print(f"[✓] 下載成功: {saved_path}")
    return saved_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public wash-hand videos for dataset expansion")
    parser.add_argument("--url", type=str, help="YouTube video URL to download")
    parser.add_argument("--name", type=str, default=None, help="Custom filename (without extension)")
    parser.add_argument("--download-recommended", action="store_true", help="Download recommended public 7-step videos")
    args = parser.parse_args()

    if args.download_recommended:
        print("[*] 開始下載推薦的公開七步洗手教學與水槽示範影片...")
        for src in RECOMMENDED_SOURCES:
            try:
                download_youtube_video(src["url"], filename=src["name"])
            except Exception as e:
                print(f"[!] 下載失敗 ({src['name']}): {e}")
    elif args.url:
        download_youtube_video(args.url, filename=args.name)
    else:
        print("請指定 --url <YouTube網址> 或使用 --download-recommended 下載推薦示範影片。")
        print("\n範例指令：")
        print("  PYTHONPATH=. ./venv/bin/python src/dataset_collector.py --url 'https://www.youtube.com/watch?v=...' --name my_wash_video")
