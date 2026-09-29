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

            left_angles  = features.get("left_angles",  [0.0] * 5)
            right_angles = features.get("right_angles", [0.0] * 5)
            left_curl  = float(np.mean(left_angles[1:]))  if has_left  else 180.0
            right_curl = float(np.mean(right_angles[1:])) if has_right else 180.0
            min_curl  = min(left_curl, right_curl)
            max_curl  = max(left_curl, right_curl)
            curl_diff = abs(left_curl - right_curl)

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

            # ── 2. [大 Thumb] ─────────────────────────────────────────────
            # 旋轉搓洗拇指：一手拇指深入對側掌心/虎口，被對側手包覆
            # 關鍵特徵：被洗手四指大張開伸展 (max_curl > 145.0 且 max_spread > 0.24)
            l_s = float(inter.get("left_four_finger_spread", 0.0))
            r_s = float(inter.get("right_four_finger_spread", 0.0))
            max_spread = max(l_s, r_s)
            is_thumb_in_web = (min_p_to_th < 0.85 or min_web_th < 0.85)
            thumb_closer_than_knuckles = (min_p_to_th <= min_kn_to_p + 0.15 or min_web_th <= min_kn_to_p + 0.15)
            has_thumb_curl = (min_curl < 135.0 or min(left_angles[0], right_angles[0]) < 120.0)
            has_spread_open_hand = (max_curl > 145.0 and max_spread > 0.24)

            if is_thumb_in_web and has_thumb_curl and palm_dist < 1.60 and scores["wrist"] <= 0.0:
                thumb_score = 3.4
                if has_spread_open_hand:
                    # 被握手四指大張開伸展，極為明確的洗大拇指姿態
                    thumb_score = 3.8
                elif thumb_closer_than_knuckles:
                    thumb_score = 3.5
                scores["thumb"] = thumb_score

            # ── 3. [弓 Knuckles] ───────────────────────────────────────────
            # 指背搓掌心：一手手指彎曲呈弓形 (min_curl < 142.0 即可)
            # 關鍵區別：若為洗大拇指 (大拇指深入虎口且被握手張開)，則不應視為弓
            not_thumb_grasp = not (is_thumb_in_web and has_spread_open_hand)
            if min_curl < 142.0 and scores["wrist"] <= 0.0 and not_thumb_grasp and palm_dist < 2.0:
                # 幾何證據打分：指節越靠近掌心越明確為弓
                knuckle_score = 3.3
                if min_kn_to_p < 0.90:
                    knuckle_score = 3.5
                if curl_diff > 20.0 or min_kn_to_p < 0.75:
                    knuckle_score += 0.2
                # 若拇指深陷虎口且無明確指節接觸掌心，抑制弓的分數
                if is_thumb_in_web and thumb_closer_than_knuckles:
                    knuckle_score = max(0.0, knuckle_score - 1.5)
                scores["knuckles"] = knuckle_score

            # ── 4. [立 Fingertips] ────────────────────────────────────────
            # 指尖搓掌心：四指指尖聚攏，垂直在對側掌心中心旋轉
            fingers_clustered = (fing_spread < 0.30)
            tips_at_palm = (min_t_to_p < 0.90 and min_t_to_p <= min_kn_to_p + 0.15)

            if fingers_clustered and tips_at_palm and palm_dist < 1.40 and not (is_thumb_in_web and has_spread_open_hand):
                tip_score = 3.5
                if scores["knuckles"] > 0.0 and min_t_to_p < min_kn_to_p:
                    tip_score += 0.2
                scores["fingertips"] = tip_score

            # ── 5. [夾 Interlace] vs [內 Inside] vs [外 Outside] ───────────
            # 雙手平掌伸展 (四指皆伸直平展 min_curl >= 142.0, max_curl > 150.0, palm_dist < 1.60)
            if min_curl >= 142.0 and max_curl > 150.0 and palm_dist < 1.60:
                l_s_val = float(inter.get("left_four_finger_spread", 99.0))
                r_s_val = float(inter.get("right_four_finger_spread", 99.0))
                mean_4_spread = (l_s_val + r_s_val) / 2.0 if l_s_val < 90 and r_s_val < 90 else min_4_spread
                wrist_palm_diff = abs(wrist_dist - palm_dist)
                thumb_dist = float(inter.get("thumb_to_thumb_dist", 99.0))

                # 【夾】(十指交錯):
                # (1) 手指交錯交替 (interlace_alts >= 3 且 x_overlap >= 0.20 且 mean_4_spread >= 0.25 且 interlace_d < 1.15)
                # (2) 雙手呈 X 形交叉且指縫交疊 (axis_angle >= 28.0 且 palm_dot < 0.10 且 x_overlap >= 0.20 且 interlace_alts >= 2 且 interlace_d < 1.25)
                # 排除雙手大拇指同向貼齊的掌心對搓 (thumb_dist < 0.50 且 palm_dot < -0.40 且 interlace_alts <= 1)
                is_interlace = (
                    (
                        (interlace_alts >= 3 and x_overlap >= 0.20 and mean_4_spread >= 0.25 and interlace_d < 1.15)
                        or (axis_angle >= 28.0 and palm_dot < 0.10 and x_overlap >= 0.20 and interlace_alts >= 2 and interlace_d < 1.25)
                    )
                    and not (thumb_dist < 0.50 and palm_dot < -0.40 and interlace_alts <= 1)
                )

                # 【內】(掌心對掌心):
                # 必須綁定「掌心面對面 (palm_dot < -0.40)」且雙手大拇指 3D 距離近 (thumb_dist < 0.55 或未指定)
                thumbs_aligned = (thumb_dist < 0.55 or thumb_dist > 90.0)
                is_inside = (
                    not is_interlace
                    and (
                        (palm_dot < -0.40 and thumbs_aligned)
                        or (palm_dot < -0.60)
                    )
                    and palm_dist < 1.35
                )

                # 【外】(掌心搓手背):
                # 依據「掌背重疊」或「法向量同向 (palm_dot > -0.20)」獨立打分
                is_outside = (
                    not is_interlace
                    and not is_inside
                    and (
                        palm_dot > -0.20
                        or (wrist_palm_diff >= 0.20 and palm_dot > -0.30 and axis_angle < 60.0)
                        or (thumb_dist > 0.70 and thumb_dist < 90.0 and palm_dot > -0.35 and axis_angle < 60.0)
                    )
                    and palm_dist < 1.50
                )

                if is_interlace:
                    scores["interlace"] = 3.6
                elif is_inside:
                    # 明確掌心相對且拇指對齊：掌心對搓 (內)
                    scores["inside"] = 3.4
                elif is_outside:
                    scores["outside"] = 1.8
                else:
                    scores["inside"] = 2.0
                    scores["outside"] = 2.0

            # 接觸微幾何增益 (Directional Contact Evidence)
            contacts = inter.get("directional_contacts", [])
            is_thumb_wrapped = (min_p_to_th < 0.75 or min_web_th < 0.75)
            if contacts and not is_flat_rubbing:
                knuckle_contact = any(c["curl"] <= 135.0 and c["knuckles_to_palm"] < 0.85
                                      and c["knuckles_to_palm"] <= c["tips_to_palm"] + 0.12 for c in contacts)
                tip_contact = any(c["curl"] >= 80.0 and c["tips_to_palm"] < 1.0
                                  and c["spread"] < 0.32
                                  and c["tips_to_palm"] < c["knuckles_to_palm"] - 0.05 for c in contacts)
                if knuckle_contact and palm_dist < 1.6 and not is_thumb_wrapped and not_thumb_grasp:
                    scores["knuckles"] = max(scores["knuckles"], 3.5)
                    # 握住手腕必要條件不滿足：指節貼在掌心，強烈抑制手腕
                    scores["wrist"] = max(0.0, scores["wrist"] - 2.0)
                if tip_contact and palm_dist < 2.0 and not is_thumb_wrapped:
                    scores["fingertips"] = max(scores["fingertips"], 3.5)
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
            has_valid_active_hand = (len(active_angles) >= 5 and any(a > 30.0 for a in active_angles))

            # [腕 Wrist — single hand]: 握持手腕必要條件：深環握且拇指尖貼近手腕
            if (
                (mean_4_angle < 115.0 and thumb_angle < 115.0 and thumb_tip_w < 0.65)
                or (mean_4_angle < 125.0 and thumb_angle < 110.0 and thumb_tip_w < 0.60)
            ):
                scores["wrist"] = 2.5

            # [大 Thumb — single hand]: 
            # 形態 A: 被握手四指在空中大張開伸展 (spread_4 > 0.28 且四指伸展 mean_4_angle > 145.0)，大拇指被包住/內收 (thumb_angle < 130.0 或 thumb_tip_w < 0.88)
            # 形態 B: 握持手四指緊握 (mean_4_angle < 125.0)，拇指閉合 (thumb_angle < 120.0)，旋轉運動中 (velocity > 0.015)
            is_all_flat = (min(active_angles) > 140.0 and thumb_angle > 135.0)
            is_single_thumb_open = (
                not is_all_flat and (
                    (spread_4 > 0.28 and max(active_angles[1], active_angles[2]) > 150.0 and (thumb_angle < 130.0 or thumb_tip_w < 0.88))
                    or (mean_4_angle > 145.0 and spread_4 > 0.24 and (thumb_angle < 125.0 or thumb_tip_w < 0.85))
                )
            )
            is_single_thumb_grip = (mean_4_angle < 125.0 and thumb_angle < 120.0 and velocity > 0.015 and thumb_tip_w >= 0.65)
            if (is_single_thumb_open or is_single_thumb_grip) and scores["wrist"] <= 0.0:
                scores["thumb"] = 2.5

            # [弓 Knuckles — single hand]: 當四指彎曲呈弓形 (45.0 <= mean_4_angle < 142.0)
            if has_valid_active_hand and 45.0 <= mean_4_angle < 142.0 and scores["wrist"] <= 0.0:
                if not is_single_thumb_open:
                    scores["knuckles"] = 2.4

            # [立 Fingertips — single hand]: 指尖聚攏 (四指指尖聚攏 pointing down/inward，且非弓形指背)
            if has_valid_active_hand and 130.0 <= mean_4_angle < 155.0 and spread_4 <= 0.22 and not is_single_thumb_open:
                scores["fingertips"] = 2.3

            # [外 Outside — single hand]: 平掌拇指內收姿態
            if has_valid_active_hand and thumb_angle < 135.0 and mean_4_angle > 140.0 and spread_4 < 0.28 and not is_single_thumb_open:
                scores["outside"] = 2.1

            # [內 Inside — single hand]: 單手平掌 (中性候選，機率約 0.32)
            if has_valid_active_hand and mean_4_angle >= 140.0 and not is_single_thumb_open and scores["outside"] <= 0.0:
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
