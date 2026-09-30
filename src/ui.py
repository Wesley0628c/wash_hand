"""
UI & Visual HUD Rendering Module
Renders modern, semi-transparent overlays, Chinese typography, and real-time wash step trackers.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List

from src.rule_classifier import LABEL_SHORT_ZH, LABEL_NAMES_ZH
from src.state_machine import STEPS_ORDER, STEPS_ZH

# Find macOS default Chinese font or fallback
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]

AVAILABLE_FONT_PATH = None
for path in FONT_CANDIDATES:
    if os.path.exists(path):
        AVAILABLE_FONT_PATH = path
        break


def get_font(size: int = 20) -> ImageFont.FreeTypeFont:
    if AVAILABLE_FONT_PATH:
        try:
            return ImageFont.truetype(AVAILABLE_FONT_PATH, size)
        except Exception:
            pass
    return ImageFont.load_default()


class WashHandHUD:
    """Renders sleek, modern, real-time action recognition UI without distracting checklist sidebars."""

    def __init__(self):
        self.font_sm = get_font(16)
        self.font_md = get_font(22)
        self.font_lg = get_font(30)
        self.font_xl = get_font(38)

    def draw_hud(
        self,
        frame: np.ndarray,
        detected_label: str,
        confidence: float,
        feedback_msg: str,
        progress_summary: Optional[Dict[str, Any]] = None,
        fps: float = 0.0,
        mode_str: str = "HYBRID",
        hands_status: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """Render clean, instant real-time HUD onto the frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # 1. Top Header Bar (Semi-transparent dark glass)
        cv2.rectangle(overlay, (0, 0), (w, 54), (18, 22, 28), -1)

        # 2. Bottom Main Action Card (Sleek Glassmorphic Card)
        banner_h = 80
        banner_y = h - banner_h - 18
        banner_x = 20
        banner_w = w - 40
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            (18, 22, 32),
            -1,
        )
        # Highlight card border based on detected label
        border_color = (40, 200, 100) if detected_label in STEPS_ORDER else (70, 80, 95)
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            border_color,
            2,
        )

        # Blend semi-transparent cards with camera image
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Convert to PIL Image for crisp Chinese typography
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw Header Info
        draw.text((25, 14), "七步洗手即時辨識系統 (Wash Hand Detect)", fill=(255, 255, 255), font=self.font_md)

        # Hand Tracking Indicator Dots in Header
        if hands_status is not None:
            left_ok = hands_status.get("left", False)
            right_ok = hands_status.get("right", False)
            left_color = (0, 255, 150) if left_ok else (130, 140, 150)
            right_color = (255, 160, 50) if right_ok else (130, 140, 150)
            status_text = f"左手 [{'●' if left_ok else '○'}]  右手 [{'●' if right_ok else '○'}]"
            draw.text((w // 2 - 80, 16), status_text, fill=(220, 230, 240), font=self.font_sm)

        draw.text((w - 240, 16), f"FPS: {fps:.1f}  |  {mode_str}", fill=(180, 205, 230), font=self.font_sm)

        # 3. Top-Center 7-Step Quick Badges Bar (Visual reference)
        badges_y = 66
        badge_w, badge_h = 60, 32
        total_badges_w = len(STEPS_ORDER) * (badge_w + 10)
        start_bx = max(20, (w - total_badges_w) // 2)

        target_step = progress_summary.get("target_step") if progress_summary else None

        for i, step in enumerate(STEPS_ORDER):
            bx = start_bx + i * (badge_w + 10)
            is_active = (detected_label == step)
            step_zh = STEPS_ZH.get(step, step)
            is_done = (progress_summary is not None and step in progress_summary.get("completed_steps", []))
            is_target = (target_step == step)

            # Badge background
            if is_done:
                bg_col = (35, 145, 75)
                txt_col = (255, 255, 255)
                outline_col = (80, 220, 120)
                badge_text = f"✓ {step_zh}"
            elif is_target and is_active:
                bg_col = (230, 160, 20)
                txt_col = (255, 255, 255)
                outline_col = (255, 230, 80)
                badge_text = f"★ {step_zh}"
            elif is_target:
                bg_col = (20, 50, 75)
                txt_col = (100, 220, 255)
                outline_col = (0, 180, 255)
                badge_text = f"▶ {step_zh}"
            elif is_active:
                bg_col = (180, 120, 25)
                txt_col = (255, 240, 200)
                outline_col = (230, 180, 60)
                badge_text = f"{step_zh}"
            else:
                bg_col = (30, 35, 45)
                txt_col = (160, 175, 190)
                outline_col = (60, 70, 85)
                badge_text = f"{step_zh}"

            draw.rectangle([bx, badges_y, bx + badge_w, badges_y + badge_h], fill=bg_col, outline=outline_col, width=2 if is_target else 1)
            draw.text((bx + (8 if (is_done or is_target) else 14), badges_y + 4), badge_text, fill=txt_col, font=self.font_sm)

        # 4. Bottom Main Action Card Content
        curr_zh = LABEL_NAMES_ZH.get(detected_label, LABEL_SHORT_ZH.get(detected_label, detected_label))
        action_color = (100, 255, 160) if detected_label in STEPS_ORDER else (200, 210, 225)
        
        target_str = f"  (目標: 【{STEPS_ZH.get(target_step, target_step)}】)" if (target_step and target_step != detected_label) else ""
        draw.text((banner_x + 22, banner_y + 12), f"當前動作：【 {curr_zh} 】{target_str}", fill=action_color, font=self.font_lg)

        # Confidence indicator in card
        if confidence > 0.0 and detected_label in STEPS_ORDER:
            conf_pct = confidence * 100.0
            conf_color = (100, 255, 160) if conf_pct >= 70.0 else (255, 200, 80) if conf_pct >= 40.0 else (180, 190, 200)
            draw.text((banner_x + banner_w - 200, banner_y + 14), f"信心度：{conf_pct:.1f}%", fill=conf_color, font=self.font_md)
        elif progress_summary is not None:
            done_cnt = progress_summary.get("completed_count", 0)
            total_cnt = progress_summary.get("total_steps", 7)
            status_summary_str = f"洗手進度：{done_cnt}/{total_cnt} 步"
            draw.text((banner_x + banner_w - 200, banner_y + 16), status_summary_str, fill=(180, 215, 255), font=self.font_md)

        # Real-time Feedback Hint
        draw.text((banner_x + 24, banner_y + 48), f"指導提示：{feedback_msg}", fill=(255, 215, 120), font=self.font_sm)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
