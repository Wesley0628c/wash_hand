"""
Rule-Based Wash Hand Classifier Module
Implements domain-rule gesture classification for the 7 steps + 1 other class,
along with multi-class probability scoring and real-time posture feedback generation.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

# Label Constants
LABELS = {
    0: "other",
    1: "inside",
    2: "outside",
    3: "interlace",
    4: "knuckles",
    5: "thumb",
    6: "fingertips",
    7: "wrist",
}

LABEL_NAMES_ZH = {
    "other": "其他 (未偵測 / 動作待確認)",
    "inside": "內 (掌心對掌心搓洗)",
    "outside": "外 (掌心搓洗手背)",
    "interlace": "夾 (十指交錯搓洗)",
    "knuckles": "弓 (指背搓洗掌心)",
    "thumb": "大 (旋轉搓洗大拇指)",
    "fingertips": "立 (指尖搓洗掌心)",
    "wrist": "腕 (旋轉搓洗手腕)",
}

LABEL_SHORT_ZH = {
    "other": "其他",
    "inside": "內",
    "outside": "外",
    "interlace": "夾",
    "knuckles": "弓",
    "thumb": "大",
    "fingertips": "立",
    "wrist": "腕",
}

FEEDBACK_ZH = {
    "other": "未偵測到手部或動作仍待確認",
    "inside": "姿勢正確：掌心對掌心搓洗",
    "outside": "姿勢正確：掌心搓洗手背",
    "interlace": "姿勢正確：十指交錯搓洗",
    "knuckles": "姿勢正確：指背搓洗掌心",
    "thumb": "姿勢正確：旋轉搓洗大拇指",
    "fingertips": "姿勢正確：指尖搓洗掌心",
    "wrist": "姿勢正確：正在旋轉搓洗手腕",
}

NAME_TO_LABEL = {v: k for k, v in LABELS.items()}


class WashHandRuleClassifier:
    """Rule-based evaluator for 7-step hand washing gestures with multi-class probability distribution."""

    def __init__(self, min_motion_velocity: float = 0.005):
        self.min_motion_velocity = min_motion_velocity

    def predict_probabilities(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Return normalized heuristic scores, not calibrated probabilities.

        Dual-hand rules use current contact geometry. Single-hand morphology
        supplies candidates; the pipeline checks motion and temporal support.
        """
        has_left = features.get("has_left", False)
        has_right = features.get("has_right", False)
        both_hands = features.get("both_hands_detected", False)

        if not (has_left or has_right):
            probs = {k: 0.01 for k in LABELS.values()}
            probs["other"] = 0.93
            return probs

        # Logit evidence scores — each class starts at 0
        # With seven zero-score competitors, softmax(3.5) ≈ 82.5%.
        scores = {k: 0.0 for k in LABELS.values()}

        # ── 1. Dual-Hand Geometric Evidence ────────────────────────────────
        if both_hands:
            inter = features.get("inter_hand", {})
            palm_dot       = float(features.get("palm_normal_dot", 0.0))
            palm_dist      = float(inter.get("palm_center_dist", 99.0))
            wrist_dist     = float(inter.get("wrist_dist", 99.0))
            min_p_to_w     = float(inter.get("min_palm_to_wrist", 99.0))
            wrist_ratio    = float(inter.get("wrist_ratio", 99.0))
            min_t_to_p     = float(inter.get("min_tips_to_palm", 99.0))
            min_t_to_w     = float(inter.get("min_tips_to_wrist", 99.0))
            min_kn_to_p    = float(inter.get("min_knuckles_to_palm", 99.0))
            min_p_to_th    = float(inter.get("min_palm_to_thumb", 99.0))
            min_web_th     = float(inter.get("min_web_to_thumb", 99.0))
            interlace_d    = float(inter.get("interlace_depth", 99.0))
            fing_spread    = float(inter.get("min_fingertip_spread", 99.0))
            tip_dist       = float(inter.get("mean_tip_dist", 99.0))
            x_overlap      = float(inter.get("finger_x_overlap", 0.0))
            min_4_spread   = float(inter.get("min_four_finger_spread", 99.0))
            interlace_alts = int(inter.get("interlace_alternations", 0))
            velocity       = float(features.get("shape_speed", features.get("velocity", 0.0) * 30.0) / 30.0)

            left_angles  = features.get("left_angles",  [0.0] * 5)
            right_angles = features.get("right_angles", [0.0] * 5)
            left_curl  = float(np.mean(left_angles[1:]))  if has_left  else 180.0
            right_curl = float(np.mean(right_angles[1:])) if has_right else 180.0
            min_curl  = min(left_curl, right_curl)
            max_curl  = max(left_curl, right_curl)
            curl_diff = abs(left_curl - right_curl)

            l_s = float(inter.get("left_four_finger_spread", 0.0))
            r_s = float(inter.get("right_four_finger_spread", 0.0))
            max_spread = max(l_s, r_s)
            mean_4_spread = (l_s + r_s) / 2.0 if l_s < 90 and r_s < 90 else min_4_spread
            thumb_dist = float(inter.get("thumb_to_thumb_dist", 99.0))
            thumb_dot  = float(inter.get("thumb_dir_dot", 0.0))

            # Hand axis alignment angle (Wrist → Middle MCP)
            left_norm_lm  = features.get("left_norm")
            right_norm_lm = features.get("right_norm")
            axis_angle = 90.0
            if left_norm_lm is not None and right_norm_lm is not None:
                v_l = left_norm_lm[9]  - left_norm_lm[0]
                v_r = right_norm_lm[9] - right_norm_lm[0]
                norm_prod = np.linalg.norm(v_l) * np.linalg.norm(v_r)
                if norm_prod > 1e-6:
                    cos_a = np.dot(v_l, v_r) / norm_prod
                    axis_angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

            # ─────────────────────────────────────────────────────────────
            # 平行特徵收集機制 (Parallel Feature Gathering)
            # 各動作獨立依據幾何證據給予分數，以最大機率 (Softmax) 決策，杜絕優先級攔截
            # ─────────────────────────────────────────────────────────────

            # ── 1. [腕 Wrist] ─────────────────────────────────────────────
            # 必要條件 (Essential Prerequisite)：
            # (1) 必須有一隻手呈現環握狀 (min_curl <= 130.0)
            # (2) 接觸核心必須是在對側手腕處：手掌緊密包覆對側手腕基底
            has_curled_grip = (min_curl <= 130.0)
            at_wrist_zone = (
                min_p_to_w < 0.68
                or wrist_dist < 0.38
                or (min_t_to_w < 0.72 and min_t_to_w < min_t_to_p - 0.08)
            )
            not_knuckle_on_palm = (min_kn_to_p > 0.65 or min_p_to_w < min_kn_to_p + 0.05)
            # 關鍵排除：接觸核心必須在手腕而非對側大拇指！若掌心緊握在大拇指上，屬於洗大拇指而非握手腕
            not_thumb_contact = not (min_p_to_th < 0.65 and min_p_to_th < min_p_to_w - 0.05)
            is_flat_rubbing = (min_curl > 135.0 and axis_angle < 65.0)

            if has_curled_grip and at_wrist_zone and not_knuckle_on_palm and not_thumb_contact and palm_dist < 2.2 and not is_flat_rubbing:
                grip_strength = 3.6
                if wrist_dist < 0.30 or min_p_to_w < 0.55:
                    grip_strength += 0.2
                scores["wrist"] = grip_strength

            # ── 2. [大 Thumb] vs [弓 Knuckles] ────────────────────────────
            # 定義平掌/伸展手 (Open Hand) 與 彎曲/包覆手 (Curled/Grip Hand) 的指向性幾何關係：
            l_4_curl = float(np.mean(left_angles[1:])) if len(left_angles) >= 5 else 180.0
            r_4_curl = float(np.mean(right_angles[1:])) if len(right_angles) >= 5 else 180.0
            l_p_to_r_th = float(inter.get("left_palm_to_right_thumb", 99.0))
            r_p_to_l_th = float(inter.get("right_palm_to_left_thumb", 99.0))
            l_kn_to_r_p = float(inter.get("left_knuckles_to_right_palm", 99.0))
            r_kn_to_l_p = float(inter.get("right_knuckles_to_left_palm", 99.0))

            if l_4_curl >= r_4_curl:
                # 左手為伸展手/被搓手，右手為包覆/彎曲手
                open_thumb_to_fist = r_p_to_l_th       # 包覆手(右) 到 被搓拇指(左)
                curled_kn_to_palm  = r_kn_to_l_p       # 彎曲手指節(右) 到 承接掌心(左)
            else:
                # 右手為伸展手/被搓手，左手為包覆/彎曲手
                open_thumb_to_fist = l_p_to_r_th       # 包覆手(左) 到 被搓拇指(右)
                curled_kn_to_palm  = l_kn_to_r_p       # 彎曲手指節(左) 到 承接掌心(右)

            # ─── (A) 計算「大 (Thumb)」證據分數 ───
            # 1. 拇指接觸分：被搓手拇指深入包覆手拳心/虎口，且目標是拇指而非手腕
            thumb_contact_score = 0.0
            if open_thumb_to_fist < 0.75 or min_web_th < 0.75 or min_p_to_th < 0.70:
                thumb_contact_score = 1.8
                if open_thumb_to_fist < 0.60 or min_web_th < 0.60:
                    thumb_contact_score = 2.2
                if open_thumb_to_fist < min_p_to_w:
                    thumb_contact_score += 0.3

            # 2. 包覆手握形分：包覆手握拳、被搓手伸展/展開
            thumb_grip_score = 0.0
            if min_curl < 135.0:
                thumb_grip_score += 0.8
                if min_curl < 120.0:
                    thumb_grip_score += 0.4
            if max_curl > 135.0:
                thumb_grip_score += 0.5
                if max_curl > 150.0:
                    thumb_grip_score += 0.3

            # 3. 動作搓動/旋轉加分
            thumb_motion_score = 0.3 if velocity > 0.010 else 0.0

            score_da = thumb_contact_score + thumb_grip_score + thumb_motion_score
            # 抑制：若指節明顯貼在掌心且拇指未深握，扣減大分數
            if (curled_kn_to_palm < 0.75 or min_kn_to_p < 0.75) and open_thumb_to_fist > 0.70:
                score_da = max(0.0, score_da - 1.5)

            # ─── (B) 計算「弓 (Knuckles)」證據分數 ───
            # 1. 指節貼掌分：指節 PIP 緊貼對側平掌掌心中央，且指節深於指尖
            knuckle_to_palm_score = 0.0
            eff_kn_dist = min(curled_kn_to_palm, min_kn_to_p)
            if eff_kn_dist < 0.85 and eff_kn_dist < min_t_to_p + 0.08:
                knuckle_to_palm_score = 1.8
                if eff_kn_dist < 0.75:
                    knuckle_to_palm_score = 2.2
                if eff_kn_dist < min_t_to_p - 0.05:
                    knuckle_to_palm_score += 0.4

            # 2. 弓手與承接手形態分：彎曲手呈弓形、承接手平掌
            gong_finger_score = 0.0
            if min_curl < 140.0:
                gong_finger_score += 0.8
            if max_curl > 140.0:
                gong_finger_score += 0.5
            if open_thumb_to_fist > 0.70:  # 伸展手拇指在空中懸空未被包住
                gong_finger_score += 0.5

            # 3. 掌心搓動分
            gong_motion_score = 0.3 if velocity > 0.010 else 0.0

            score_gong = knuckle_to_palm_score + gong_finger_score + gong_motion_score
            # 抑制：若大拇指被深握且指節遠離掌心，扣減弓分數
            if (open_thumb_to_fist < 0.65 or min_web_th < 0.65) and eff_kn_dist > open_thumb_to_fist + 0.08:
                score_gong = max(0.0, score_gong - 1.5)

            # 抑制：雙手平掌對搓 (內) 或平掌貼手背 (外) 時，禁止誤判為弓
            is_opposing_palms_flat = (palm_dot < -0.35 and min_curl > 125.0 and curl_diff < 22.0)
            is_dorsum_overlay_flat = (thumb_dist > 0.70 and thumb_dot < -0.10 and min_curl > 125.0 and curl_diff < 22.0)
            if is_opposing_palms_flat or is_dorsum_overlay_flat or (min_curl > 132.0 and curl_diff < 18.0):
                score_gong = 0.0

            # ─── (C) 計算「立 (Fingertips)」證據分數 ───
            # 1. 指尖聚攏分：四指指尖聚集成束 (放寬視角門檻至 0.38)
            li_cluster_score = 0.0
            if fing_spread < 0.38:
                li_cluster_score = 1.8
                if fing_spread < 0.28:
                    li_cluster_score = 2.3

            # 2. 指尖深於指節 (Tips on Palm Depth) —— 關鍵鑑別點！
            li_depth_score = 0.0
            if min_t_to_p < 0.95 and min_t_to_p <= eff_kn_dist + 0.12:
                li_depth_score = 1.0
                if min_t_to_p < eff_kn_dist - 0.05:
                    li_depth_score = 1.5
                if min_t_to_p < 0.70:
                    li_depth_score += 0.3

            # 3. 連續搓動/旋轉分
            li_motion_score = 0.3 if velocity > 0.010 else 0.0

            score_li = li_cluster_score + li_depth_score + li_motion_score
            if palm_dist >= 1.60 or open_thumb_to_fist < 0.60:
                score_li = max(0.0, score_li - 1.5)

            # 相互抑制：若指尖明顯聚攏且比指節更靠近掌心，扣減弓分數
            if fing_spread < 0.35 and min_t_to_p < eff_kn_dist:
                score_gong = max(0.0, score_gong - 1.8)

            # ─── (D) 綜合三者競爭裁決 (Thumb vs Knuckles vs Fingertips) ───
            is_thumb_wrapped = False
            is_knuckle_on_palm = False

            if scores["wrist"] <= 0.0 and max(score_da, score_gong, score_li) >= 2.8:
                if score_li >= score_da and score_li >= score_gong:
                    scores["fingertips"] = min(3.8, score_li)
                elif score_da > score_gong:
                    scores["thumb"] = min(3.8, score_da)
                    is_thumb_wrapped = True
                elif score_gong > 0.0:
                    scores["knuckles"] = min(3.8, score_gong)
                    is_knuckle_on_palm = True

            # ── 5. [夾 Interlace] vs [內 Inside] vs [外 Outside] ───────────
            # 雙手平掌伸展 (放寬至 min_curl >= 120.0, max_curl > 135.0, palm_dist < 1.60)
            if min_curl >= 120.0 and max_curl > 135.0 and palm_dist < 1.60:
                l_s_val = float(inter.get("left_four_finger_spread", 99.0))
                r_s_val = float(inter.get("right_four_finger_spread", 99.0))
                mean_4_spread = (l_s_val + r_s_val) / 2.0 if l_s_val < 90 and r_s_val < 90 else min_4_spread
                wrist_palm_diff = abs(wrist_dist - palm_dist)
                thumb_dist = float(inter.get("thumb_to_thumb_dist", 99.0))
                thumb_dot  = float(inter.get("thumb_dir_dot", 0.0))

                is_interlace = (interlace_alts >= 4 and x_overlap >= 0.25
                                and mean_4_spread >= 0.30
                                and axis_angle > 28.0 and interlace_d < 1.15)

                # 大拇指同邊 vs 不同邊判定準則:
                # 內 (掌心相對): 兩手鏡像對稱 -> 大拇指對大拇指 (同邊: thumb_dot > 0.0 或 thumb_dist < 0.65)
                # 外 (掌心覆蓋手背): 兩手同向疊合 -> 大拇指不同邊 (異邊: thumb_dot < -0.10 且 thumb_dist > 0.70)
                thumbs_same_side = (thumb_dot > 0.0 or thumb_dist < 0.65)
                thumbs_opposite_side = (thumb_dot < -0.10 and thumb_dist > 0.70)

                is_inside = (not is_interlace and (
                    thumbs_same_side or (palm_dot < -0.30 and not thumbs_opposite_side)
                ) and palm_dist < 1.35)

                is_outside = (not is_interlace and not is_inside and (
                    thumbs_opposite_side
                    or ((palm_dot > -0.20 or wrist_palm_diff >= 0.20 or axis_angle >= 28.0) and axis_angle < 75.0)
                ) and palm_dist < 1.50)

                if is_interlace:
                    scores["interlace"] = 3.6
                elif thumbs_opposite_side and not is_interlace:
                    # 明確大拇指異邊：掌心覆蓋手背 (外)
                    scores["outside"] = 3.6
                    scores["knuckles"] = max(0.0, scores["knuckles"] - 2.0)
                elif is_inside:
                    # 明確大拇指同邊或掌心相對：掌心對搓 (內)
                    scores["inside"] = 3.6
                    scores["knuckles"] = max(0.0, scores["knuckles"] - 2.0)
                elif is_outside:
                    scores["outside"] = 1.8
                else:
                    scores["inside"] = 2.0
                    scores["outside"] = 2.0

            # 接觸微幾何增益 (Directional Contact Evidence)
            contacts = inter.get("directional_contacts", [])
            is_thumb_wrapped = (min_p_to_th < 0.75 or min_web_th < 0.75)
            if contacts and not is_flat_rubbing and scores["inside"] <= 0.0 and scores["outside"] <= 0.0:
                knuckle_contact = any(c["curl"] <= 135.0 and c["knuckles_to_palm"] < 0.85
                                      and c["knuckles_to_palm"] <= c["tips_to_palm"] + 0.12 for c in contacts)
                tip_contact = any(c["curl"] >= 80.0 and c["tips_to_palm"] < 1.0
                                  and c["spread"] < 0.32
                                  and c["tips_to_palm"] < c["knuckles_to_palm"] - 0.05 for c in contacts)
                if knuckle_contact and palm_dist < 1.6 and not is_thumb_wrapped and not tip_contact and scores["fingertips"] <= 0.0:
                    scores["knuckles"] = max(scores["knuckles"], 3.5)
                    # 握住手腕必要條件不滿足：指節貼在掌心，強烈抑制手腕
                    scores["wrist"] = max(0.0, scores["wrist"] - 2.0)
                if tip_contact and palm_dist < 2.0 and not is_thumb_wrapped:
                    scores["fingertips"] = max(scores["fingertips"], 3.6)
                    # 指尖貼在掌心，強烈抑制弓與手腕
                    scores["knuckles"] = max(0.0, scores["knuckles"] - 1.5)
                    scores["wrist"] = max(0.0, scores["wrist"] - 2.0)
                if knuckle_contact and not tip_contact and not is_thumb_wrapped:
                    scores["thumb"] = max(0.0, scores["thumb"] - 1.0)
                if tip_contact and not knuckle_contact:
                    if not is_thumb_wrapped:
                        scores["thumb"] = max(0.0, scores["thumb"] - 1.0)
                    scores["interlace"] = max(0.0, scores["interlace"] - 1.0)
                if knuckle_contact or tip_contact:
                    scores["other"] = 0.0

            if max(scores.values()) <= 0.0:
                scores["other"] = 2.0

        # ── 2. Single-Hand / Merged Cluster Fallback ────────────────────────
        else:
            active_angles = features.get("active_angles", [0.0] * 5)
            active_spread = float(features.get("active_spread", 0.0))
            spread_4      = float(features.get("active_four_finger_spread", 0.0))
            velocity      = float(features.get("shape_speed", features.get("velocity", 0.0) * 30.0) / 30.0)
            mean_4_angle  = float(np.mean(active_angles[1:])) if len(active_angles) >= 5 else 180.0
            thumb_angle   = float(active_angles[0])           if len(active_angles) >= 1 else 180.0
            thumb_tip_w   = float(features.get("active_thumb_tip_to_wrist", 2.0))

            # [腕 Wrist — single hand]: 握持手腕必要條件：深環握且拇指尖貼近手腕
            if (
                (mean_4_angle < 115.0 and thumb_angle < 115.0 and thumb_tip_w < 0.65)
                or (mean_4_angle < 125.0 and thumb_angle < 110.0 and thumb_tip_w < 0.60)
            ):
                scores["wrist"] = 2.5

            # [大 Thumb — single hand]: 
            # 形態 A: 被握手四指在空中張開 (spread_4 > 0.30 且食/中指伸展)，但排除全手指均勻平展的掌心對搓候選
            # 形態 B: 握持手四指緊握 (mean_4_angle < 125.0)，拇指閉合 (thumb_angle < 120.0)，旋轉運動中 (velocity > 0.015)
            is_all_flat = (min(active_angles) > 145.0)
            is_single_thumb_open = (
                not is_all_flat and (
                    (spread_4 > 0.30 and max(active_angles[1], active_angles[2]) > 155.0 and min(active_angles[1:]) < 130.0)
                    or (mean_4_angle > 145.0 and spread_4 > 0.24 and (thumb_angle < 130.0 or thumb_tip_w < 0.88))
                )
            )
            is_single_thumb_grip = (mean_4_angle < 125.0 and thumb_angle < 120.0 and velocity > 0.015 and thumb_tip_w >= 0.65)
            if (is_single_thumb_open or is_single_thumb_grip) and scores["wrist"] <= 0.0:
                scores["thumb"] = 2.5

            # [弓 Knuckles — single hand]: 當四指彎曲呈弓形 (45.0 <= mean_4_angle < 142.0) 即可判定為弓
            has_valid_active_hand = (len(active_angles) >= 5 and any(a > 30.0 for a in active_angles))
            if has_valid_active_hand and 45.0 <= mean_4_angle < 142.0 and scores["wrist"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["knuckles"] = 2.4

            # [立 Fingertips — single hand]: 指尖聚攏 (四指指尖聚攏 pointing down/inward，且非弓形指背)
            if has_valid_active_hand and 130.0 <= mean_4_angle < 155.0 and spread_4 <= 0.22 and scores["knuckles"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["fingertips"] = 2.3

            # [外 Outside — single hand]: 平掌拇指內收姿態
            if has_valid_active_hand and thumb_angle < 135.0 and mean_4_angle > 140.0 and spread_4 < 0.28 and scores["thumb"] <= 0.0:
                scores["outside"] = 2.1

            # [內 Inside — single hand]: 單手平掌 (中性候選，機率約 0.32)
            if has_valid_active_hand and mean_4_angle >= 140.0 and scores["outside"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["inside"] = 1.2

            if max(scores.values()) <= 0.0:
                scores["other"] = 2.0

        # A stale second hand cannot upgrade a single visible palm into strong
        # palm-to-palm evidence. Retain a modest display candidate using recent
        # geometry, below the credit threshold; no completion-history prior.
        if features.get("contains_held", False) and not features.get("both_hands_observed", False):
            scores["inside"] = min(scores["inside"], 1.6)

        # Score dorsum rubbing independently so a noisy opposing normal cannot
        # pre-empt it in the inside/outside if-elif chain. Tracker evidence has
        # already checked region, sliding, roles, and competing contact types.
        back = features.get("back_evidence", {})
        if back.get("score", 0.0) >= 0.60:
            back_score = 3.4
            scores["outside"] = max(scores["outside"], back_score)
            if back_score >= scores["inside"] - 0.3:
                scores["inside"] -= 0.8

        # ── Softmax normalization ───────────────────────────────────────────
        score_arr = np.array(list(scores.values()), dtype=np.float32)
        score_arr = np.clip(score_arr, -10.0, 15.0)
        exp_scores = np.exp(score_arr - np.max(score_arr))
        sum_exp = float(np.sum(exp_scores))
        norm_probs = exp_scores / max(1e-6, sum_exp)

        return {act: float(prob) for act, prob in zip(scores.keys(), norm_probs)}



    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Evaluate extracted features and return:
          (label_name, confidence, feedback_message)
        """
        probs = self.predict_probabilities(features)
        top_action = max(probs, key=probs.get)
        top_conf = probs[top_action]
        feedback = FEEDBACK_ZH.get(top_action, "動作調整中，請依步驟搓洗")
        return top_action, top_conf, feedback
