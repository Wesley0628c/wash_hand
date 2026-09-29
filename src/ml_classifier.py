"""
Machine Learning (XGBoost / LightGBM) Wash Hand Gesture Classifier Module
Loads trained model and integrates 15-frame rolling statistical features
to output calibrated multi-class probabilities.
"""

import os
import joblib
from typing import Dict, Any, Tuple, Optional
import numpy as np

from src.features import feature_dict_to_vector
from src.temporal_features import TemporalFeatureBuffer
from src.rule_classifier import LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH, FEEDBACK_ZH, WashHandRuleClassifier


class WashHandMLClassifier:
    """XGBoost / Temporal ML Classifier for 7-step Hand Wash recognition."""

    def __init__(
        self,
        model_path: str = "models/wash_hand_xgb.joblib",
        buffer_size: int = 15,
        hybrid_with_rules: bool = True,
        rule_weight: float = 0.25,
    ):
        self.model_path = model_path
        self.buffer = TemporalFeatureBuffer(buffer_size=buffer_size)
        self.hybrid_with_rules = hybrid_with_rules
        self.rule_weight = rule_weight
        self.rule_classifier = WashHandRuleClassifier() if hybrid_with_rules else None

        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"[+] 成功載入 ML 模型: {model_path}")
            except Exception as e:
                print(f"[!] Warning: Failed to load model from {model_path}: {e}")
        else:
            print(f"[!] Warning: Model {model_path} not found. Fallback to Rule-based classifier.")

    def reset(self):
        """Reset temporal buffer."""
        self.buffer.reset()

    def predict_probabilities(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate calibrated class probabilities from extracted features using XGBoost + 15-frame history.
        """
        has_left = features.get("has_left", False)
        has_right = features.get("has_right", False)

        if not (has_left or has_right):
            self.buffer.reset()
            probs = {k: 0.01 for k in LABELS.values()}
            probs["other"] = 0.93
            return probs

        # 1. Extract 160-dim feature vector and update 15-frame buffer
        vec_160 = feature_dict_to_vector(features)
        temp_960 = self.buffer.update(vec_160)

        # 2. ML Prediction
        if self.model is not None:
            try:
                # Shape: (1, 960)
                ml_probs_arr = self.model.predict_proba(temp_960.reshape(1, -1))[0]
                if hasattr(self.model, "classes_"):
                    ml_probs = {k: 0.001 for k in LABELS.values()}
                    for cls_idx, prob in zip(self.model.classes_, ml_probs_arr):
                        if cls_idx in LABELS:
                            ml_probs[LABELS[cls_idx]] = float(prob)
                else:
                    ml_probs = {LABELS[i]: float(ml_probs_arr[i]) for i in range(len(ml_probs_arr))}
            except Exception as e:
                if self.rule_classifier is not None:
                    ml_probs = self.rule_classifier.predict_probabilities(features)
                else:
                    ml_probs = {k: 0.01 for k in LABELS.values()}
                    ml_probs["other"] = 0.93
        else:
            if self.rule_classifier is not None:
                ml_probs = self.rule_classifier.predict_probabilities(features)
            else:
                ml_probs = {k: 0.01 for k in LABELS.values()}
                ml_probs["other"] = 0.93

        # 3. Hybrid Blending with Geometric Rules (if enabled)
        if self.hybrid_with_rules and self.rule_classifier is not None and self.model is not None:
            rule_probs = self.rule_classifier.predict_probabilities(features)
            blended_probs = {}
            w_rule = self.rule_weight
            w_ml = 1.0 - w_rule
            for k in LABELS.values():
                blended_probs[k] = w_ml * ml_probs.get(k, 0.0) + w_rule * rule_probs.get(k, 0.0)

            # Re-normalize
            total = sum(blended_probs.values())
            for k in blended_probs:
                blended_probs[k] /= max(1e-6, total)
            return blended_probs

        return ml_probs

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Evaluate features and return: (label_name, confidence, feedback_message)
        """
        probs = self.predict_probabilities(features)
        top_action = max(probs, key=probs.get)
        top_conf = probs[top_action]
        feedback = FEEDBACK_ZH.get(top_action, "動作調整中，請依步驟搓洗")
        return top_action, top_conf, feedback
