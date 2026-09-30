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

            # 抑制：外（掌心搓手背）時，被壓手拇指自然靠近壓手 → 不應觸發大
            # 拇指異邊（thumb_dot < -0.15 & thumb_dist > 0.72）是外的強信號
            _th_dot_b2  = float(inter.get("thumb_dir_dot", 0.0))
            _th_dist_b2 = float(inter.get("thumb_to_thumb_dist", 99.0))
            _thumbs_opp_b2 = (_th_dot_b2 < -0.15 and _th_dist_b2 > 0.72)
            if _thumbs_opp_b2 and palm_dist < 1.50 and min_curl > 118.0:
                # 外的手型（拇指異邊 + 雙手平掌），非真正握拇指動作
                score_da = max(0.0, score_da - 2.0)

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

            # ── 5. [夾] vs [外] vs [內]: 依主要接觸位置與運動不對稱性評分 ───────────
            motion_asym = float(inter.get("motion_asymmetry", 0.0))
            left_spd    = float(inter.get("left_speed", 0.0))
            right_spd   = float(inter.get("right_speed", 0.0))
            max_spd     = max(left_spd, right_spd)
            min_spd     = min(left_spd, right_spd)
            # 做「外」時，上方手掌在手背上搓動 (max_spd > 0.02)，下方手相對穩定充當支撐 (min_spd < 0.04)
            is_asymmetric_rub = (motion_asym > 0.35 and max_spd > 0.018 and min_spd < 0.040)

            # 放寬手指微屈門檻：支援平掌與微屈壓手搓手背 (min_curl >= 105.0)
            if (min_curl >= 105.0 and max_curl > 130.0 and palm_dist < 1.65) or (is_asymmetric_rub and palm_dist < 1.60):
                l_s_val = float(inter.get("left_four_finger_spread", 99.0))
                r_s_val = float(inter.get("right_four_finger_spread", 99.0))
                mean_4_spread   = (l_s_val + r_s_val) / 2.0 if l_s_val < 90 and r_s_val < 90 else min_4_spread
                wrist_palm_diff = abs(wrist_dist - palm_dist)
                thumb_dist      = float(inter.get("thumb_to_thumb_dist", 99.0))
                thumb_dot       = float(inter.get("thumb_dir_dot", 0.0))
                l2r_d           = float(inter.get("left_to_right_interlace_depth", 99.0))
                r2l_d           = float(inter.get("right_to_left_interlace_depth", 99.0))
                close_pairs     = int(inter.get("cross_finger_close_pairs", 0))
                overlap_ratio   = max(float(inter.get("interlace_overlap_ratio", 0.0)), float(inter.get("finger_x_overlap", 0.0)))

                # 拇指同邊 / 異邊
                thumbs_same_side     = (thumb_dot > 0.0 or thumb_dist < 0.60)
                thumbs_opposite_side = (thumb_dot < -0.15 and thumb_dist > 0.72)

                # 非對稱度：外 特有（一手指尖貼另一手手背，另方向遠）
                depth_asym = abs(l2r_d - r2l_d)
                min_one_d  = min(l2r_d, r2l_d)

                # 手指真正交錯的必要條件：交替投影 + 兩手有角度 + 指尖深入到對方指根
                # axis_angle > 15°：內時兩手鏡像平行（≈ 0~10°），夾時兩手有交叉角度（≈ 20~80°）
                # interlace_d < 1.05：指尖確實接近對方指根（比舊版 < 1.0 稍寬鬆以涵蓋更多夾的姿勢）
                fingers_truly_interlaced = (
                    interlace_alts >= 2
                    and overlap_ratio >= 0.35
                    and axis_angle > 15.0        # 從 25° 放寬至 15°，仍可排除內（0~10°）
                    and interlace_d < 1.05       # 從 0.90 放寬至 1.05（涵蓋更多真實夾的姿勢）
                )

                # ═══ 夾 score ═════════════════════════════════════════════
                # 原則：close_pairs 只加少量分，真正的交錯才加大分
                s_jia = 0.0
                if palm_dist < 1.20:
                    s_jia += 0.5                          # 兩手靠近（低分，只是前提）
                if close_pairs >= 4:
                    s_jia += 0.8                          # 多根手指靠近（只代表「近」）
                if close_pairs >= 8:
                    s_jia += 0.5                          # 更多靠近（仍只是「近」）
                if fingers_truly_interlaced:
                    s_jia += 3.0                          # 手指確實交錯（核心強信號）
                if fingers_truly_interlaced and close_pairs >= 6:
                    s_jia += 1.5                          # 交錯 + 大量接觸 ≈ 真正夾
                if interlace_alts >= 3 and fingers_truly_interlaced:
                    s_jia += 0.5                          # 高度交錯
                # axis_angle 高（兩手 X 交叉）+ 手指交替 → 夾的強增益
                if axis_angle > 35.0 and interlace_alts >= 2:
                    s_jia += 2.0                          # X 交叉 + 交替 = 夾的特徵
                if axis_angle > 50.0 and interlace_alts >= 1:
                    s_jia += 1.0                          # 更大角度的 X 交叉
                # 抑制：明確是內（掌心相對 + 無交錯）→ 強力扣分
                if palm_dot < -0.25 and palm_dist < 1.20 and interlace_alts < 2:
                    s_jia -= 3.0
                # 抑制：外的信號（低角度 + 拇指異邊）→ X 交叉時不扣（那是夾）
                if depth_asym > 0.25 and thumbs_opposite_side and axis_angle < 30.0:
                    s_jia -= 2.5
                # 抑制：拇指同邊是內的強信號
                if thumbs_same_side and palm_dot < -0.35:
                    s_jia -= 2.5
                # 抑制：非對稱接觸偏強（只在低角度時扣，高角度 X 交叉本來就不對稱）
                if depth_asym > 0.40 and axis_angle < 30.0:
                    s_jia -= 1.5
                # 抑制：運動不對稱 (外) 時非夾
                if is_asymmetric_rub and axis_angle < 35.0:
                    s_jia -= 2.0
                # 抑制：拇指主導 → 大
                if thumb_dist < 0.45:
                    s_jia -= 2.5
                # 抑制：指尖壓掌心 → 立/弓
                if min_t_to_p < 0.50:
                    s_jia -= 2.0

                # ═══ 外 score ═════════════════════════════════════════════
                # 判斷重點：一動一靜運動不對稱 + 一手指尖靠近另一手指根（不對稱）+ 拇指異邊/掌心同向 + 低交叉角度
                s_wai = 0.0
                if is_asymmetric_rub:
                    s_wai += 2.8                          # 上方手搓動、下方手相對靜止 (外核心動態特徵)
                if min_one_d < 0.75:
                    s_wai += 1.5                          # 一手指尖靠近另一手指根
                if min_one_d < 0.55:
                    s_wai += 0.5
                if depth_asym > 0.25:
                    s_wai += 1.5                          # 非對稱接觸（外核心特徵）
                if depth_asym > 0.50:
                    s_wai += 1.0                          # 更強的非對稱
                # 掌心同向朝向 (一掌面覆蓋一手背)
                if palm_dot > 0.25:
                    if is_asymmetric_rub:
                        s_wai += 2.5                      # 掌心同向 + 運動不對稱滑動 → 強外
                    else:
                        s_wai += 1.2                      # 僅靜態同向無滑動 → 弱外候選
                # 拇指異邊：只在低 axis_angle 時給滿分（外＝低角度疊合，夾＝高角度交叉）
                if thumbs_opposite_side:
                    if axis_angle < 30.0:                 # 低角度 → 確實是外
                        s_wai += 2.5
                    elif axis_angle < 45.0:               # 中角度 → 可能是夾或外
                        s_wai += 1.2
                    else:                                 # 高角度（X 交叉）→ 更像夾
                        s_wai += 0.3
                elif not thumbs_same_side:
                    s_wai += 0.5
                if interlace_alts < 2 or overlap_ratio < 0.30:
                    s_wai += 1.0                          # 手指沒有明顯交錯
                # 抑制：高 axis_angle（X 交叉）→ 傾向夾，不是外
                if axis_angle > 45.0:
                    s_wai -= 1.5
                if axis_angle > 60.0:
                    s_wai -= 1.0                          # 更強的 X 交叉
                # 抑制：手指確實交錯 → 夾，不是外
                if fingers_truly_interlaced:
                    s_wai -= 2.5
                # 抑制：掌心相對且強 → 不是外
                if palm_dot < -0.50 and thumbs_same_side:
                    s_wai -= 2.0
                # 抑制：拇指同邊 → 不是外
                if thumbs_same_side and palm_dot < -0.20:
                    s_wai -= 1.5

                # ═══ 內 score ═════════════════════════════════════════════
                # 判斷重點：掌心對掌心（對稱）+ 拇指同邊 + 無明顯交錯
                s_nei = 0.0
                if palm_dist < 1.20 and palm_dot < -0.25:
                    s_nei += 3.0                          # 掌心相對靠近（核心強信號）
                if thumbs_same_side and palm_dot < -0.10:
                    s_nei += 2.5                          # 拇指同邊 + 掌心相對
                if interlace_alts < 2:
                    s_nei += 1.5                          # 無明顯交錯（重要正向加分）
                if depth_asym < 0.20:
                    s_nei += 0.5                          # 對稱接觸
                if close_pairs < 4:
                    s_nei += 0.5                          # 指尖未互相插入
                # 抑制：運動不對稱 (外) → 內要求雙手對稱同動
                if is_asymmetric_rub:
                    s_nei -= 2.5
                # 抑制：掌心同向 (一掌蓋一背) → 絕非內
                if palm_dot > 0.20:
                    s_nei -= 3.0
                # 抑制：拇指異邊 → 不是內
                if thumbs_opposite_side:
                    s_nei -= 2.5
                # 抑制：手指確實交錯 → 不是內
                if fingers_truly_interlaced:
                    s_nei -= 3.0
                # 抑制：非對稱明顯 → 不是內
                if depth_asym > 0.40:
                    s_nei -= 2.0

                # ═══ 優先規則 + 最高分勝出 ════════════════════════════════
                SCORE_MIN = 2.0

                # 優先判斷：明確內（掌心相對強 + 完全無交錯）
                clear_inside  = (s_nei >= 5.0 and interlace_alts < 2 and palm_dot < -0.25)
                # 優先判斷：明確外（拇指異邊強或運動不對稱強 + 完全無交錯）
                clear_outside = ((s_wai >= 4.5 or (is_asymmetric_rub and s_wai >= 3.5)) and interlace_alts < 2)
                # 夾必要條件：一定要有真正交錯的證據
                # 優先以 fingers_truly_interlaced 判斷；次選：中度交錯（alts + ratio + 足夠配對）
                jia_qualified = (
                    fingers_truly_interlaced
                    or (interlace_alts >= 2 and overlap_ratio >= 0.35 and close_pairs >= 5 and axis_angle > 10.0)
                    or (interlace_alts >= 3 and close_pairs >= 6)
                )

                if clear_inside:
                    scores["inside"] = min(3.8, 2.0 + s_nei * 0.20)
                    scores["knuckles"] = max(0.0, scores["knuckles"] - 1.5)
                elif clear_outside:
                    scores["outside"] = min(3.8, 2.0 + s_wai * 0.20)
                    scores["knuckles"] = max(0.0, scores["knuckles"] - 1.5)
                elif jia_qualified and s_jia >= 3.0 and s_jia >= s_wai and s_jia >= s_nei:
                    scores["interlace"] = min(3.8, 2.0 + s_jia * 0.25)
                elif max(s_wai, s_nei) >= SCORE_MIN:
                    if s_wai >= s_nei:
                        scores["outside"] = min(3.8, 1.8 + (s_wai - 2.0) * 0.20)
                        scores["knuckles"] = max(0.0, scores["knuckles"] - 1.5)
                    else:
                        scores["inside"] = min(3.8, 2.0 + s_nei * 0.25)
                        scores["knuckles"] = max(0.0, scores["knuckles"] - 1.5)
                else:
                    # 無明確信號，給低分候選
                    if palm_dot > 0.20 and palm_dist < 1.40:
                        scores["outside"] = 1.7
                        scores["inside"]  = 1.3
                    elif palm_dist < 1.35:
                        scores["inside"]  = max(scores["inside"],  1.5)
                        scores["outside"] = max(scores["outside"], 1.5)

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

            # [大 Thumb — single hand]: 大（旋轉搓大拇指）需要雙手，單手偵測時不可能成立，不作判斷
            # is_single_thumb_grip = False  ← 永遠不觸發

            # [弓 Knuckles — single hand]: 當四指彎曲呈弓形 (45.0 <= mean_4_angle < 135.0) 且非握持大拇指
            has_valid_active_hand = (len(active_angles) >= 5 and any(a > 30.0 for a in active_angles))
            if has_valid_active_hand and 45.0 <= mean_4_angle < 135.0 and scores["wrist"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["knuckles"] = 2.4

            # [立 Fingertips — single hand]: 指尖聚攏 (四指指尖聚攏 pointing down/inward，且非弓形指背)
            if has_valid_active_hand and 130.0 <= mean_4_angle < 155.0 and spread_4 <= 0.22 and scores["knuckles"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["fingertips"] = 2.3

            # [外 Outside / 內 Inside — single hand]: 單手平掌 (中性候選，由時序追蹤器與手背特徵裁決)
            if has_valid_active_hand and mean_4_angle >= 135.0 and scores["wrist"] <= 0.0 and scores["thumb"] <= 0.0:
                scores["inside"] = 1.6
                scores["outside"] = 1.2

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
