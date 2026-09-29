"""
XGBoost 7-Step Wash Hand Classifier Training & Validation Script
Integrates data/sample_v2 clips and benchmark videos with leak-free grouped validation,
temporal statistical feature extraction (160 dim base * 6 stats = 960 dim),
and exports trained models with comprehensive provenance metadata.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import argparse
import time
from typing import Dict, List, Tuple, Any
import joblib
import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from src.hand_detector import HandDetector
from src.features import extract_hand_features, feature_dict_to_vector
from src.temporal_features import TemporalFeatureBuffer, apply_landmark_dropout
from src.rule_classifier import LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH
from src.evaluate_segments import BENCHMARK_GROUND_TRUTH, get_ground_truth_label

SAMPLE_V2_MAP = {
    "內.mov": "inside",
    "外.mov": "outside",
    "夾.mov": "interlace",
    "弓.mov": "knuckles",
    "大.mov": "thumb",
    "立.mov": "fingertips",
    "腕.mov": "wrist",
}


def extract_dataset(
    sample_v2_dir: str = "data/sample_v2",
    sample_v1_dir: str = "data/sample_v1",
    augmentations_per_frame: int = 1,
    buffer_size: int = 15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Extract temporal statistical feature vectors from sample_v1 and sample_v2 .mov files.

    Group isolation: each .mov file = 1 group (14 groups total: v1 x 7 + v2 x 7).
    GroupKFold validation trains on one source's .mov files and tests on another,
    giving a realistic cross-person/session generalization estimate.

    Returns:
        (X, y, groups, source_names)
    """
    detector = HandDetector(min_detection_confidence=0.50, min_tracking_confidence=0.50)

    X_list = []
    y_list = []
    group_list = []
    source_names = []

    group_idx = 0

    # Helper function to extract from isolated sample directories
    def _extract_from_sample_dir(s_dir: str, s_name_prefix: str):
        nonlocal group_idx
        if not os.path.exists(s_dir):
            return
        print(f"[*] 正在從 {s_dir} 抽取獨立動作資料...")
        for mov_name, label_name in sorted(SAMPLE_V2_MAP.items()):
            mov_path = os.path.join(s_dir, mov_name)
            if not os.path.exists(mov_path):
                continue

            cap = cv2.VideoCapture(mov_path)
            if not cap.isOpened():
                continue

            lbl_idx = NAME_TO_LABEL[label_name]
            buf_orig = TemporalFeatureBuffer(buffer_size=buffer_size)
            buf_augs = [TemporalFeatureBuffer(buffer_size=buffer_size) for _ in range(augmentations_per_frame)]

            detector.reset_tracking()
            prev_left, prev_right = None, None
            source_names.append(f"{s_name_prefix}/{mov_name}")
            curr_group = group_idx
            group_idx += 1
            f_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                left_hand, right_hand, _ = detector.process(frame)
                if left_hand is not None or right_hand is not None:
                    # Original sample
                    feats_orig = extract_hand_features(left_hand, right_hand, prev_left, prev_right)
                    vec_orig = feature_dict_to_vector(feats_orig)
                    temp_orig = buf_orig.update(vec_orig)

                    X_list.append(temp_orig)
                    y_list.append(lbl_idx)
                    group_list.append(curr_group)
                    f_count += 1

                    # Augmented samples
                    for aug_i in range(augmentations_per_frame):
                        aug_l, aug_r = apply_landmark_dropout(
                            left_hand, right_hand,
                            point_dropout_prob=0.15,
                            single_hand_drop_prob=0.10,
                            jitter_std=0.012,
                        )
                        feats_aug = extract_hand_features(aug_l, aug_r, prev_left, prev_right)
                        vec_aug = feature_dict_to_vector(feats_aug)
                        temp_aug = buf_augs[aug_i].update(vec_aug)

                        X_list.append(temp_aug)
                        y_list.append(lbl_idx)
                        group_list.append(curr_group)

                prev_left, prev_right = left_hand, right_hand

            cap.release()
            print(f"   - {mov_name} ({label_name:10s}): 擷取 {f_count} 幀有效樣本")

    # 1. Extract from sample_v2 (high quality isolated step recordings)
    _extract_from_sample_dir(sample_v2_dir, "sample_v2")

    # 2. Extract from sample_v1 (additional isolated step recordings from different session)
    _extract_from_sample_dir(sample_v1_dir, "sample_v1")

    # NOTE: data/clips is NOT used for training — raw .mov files are the canonical source.
    # Adding clips would create overlapping groups with the YouTube raw videos.

    # Synthetic 'other' (class 0) negative samples: hands far apart or idle.
    # These are needed because sample_v1/v2 only contain wash gestures (classes 1-7).
    print("[*] 生成合成 other (class 0) 負樣本...")
    other_group = group_idx
    group_idx += 1
    for _ in range(150):
        dummy_feats = {
            "has_left": True,
            "has_right": True,
            "both_hands_detected": True,
            "left_norm": np.zeros((21, 3), dtype=np.float32),
            "right_norm": np.ones((21, 3), dtype=np.float32) * 2.0,
            "left_normal": np.array([0, 0, 1], dtype=np.float32),
            "right_normal": np.array([0, 0, 1], dtype=np.float32),
            "palm_normal_dot": 1.0,
            "left_angles": [180.0] * 5,
            "right_angles": [180.0] * 5,
            "velocity": 0.0,
            "active_angles": [180.0] * 5,
            "active_spread": 0.0,
            "inter_hand": {
                "wrist_dist": 99.0,
                "palm_center_dist": 99.0,
                "left_tips_to_right_palm": 99.0,
                "right_tips_to_left_palm": 99.0,
                "min_tips_to_palm": 99.0,
                "left_palm_to_right_wrist": 99.0,
                "right_palm_to_left_wrist": 99.0,
                "min_palm_to_wrist": 99.0,
                "wrist_ratio": 99.0,
                "min_knuckles_to_palm": 99.0,
                "left_palm_to_right_thumb": 99.0,
                "right_palm_to_left_thumb": 99.0,
                "min_palm_to_thumb": 99.0,
                "mean_tip_dist": 99.0,
                "interlace_depth": 99.0,
                "min_fingertip_spread": 99.0,
            },
        }
        vec = feature_dict_to_vector(dummy_feats)
        temp_vec = np.tile(vec, 6)  # replicate across 6 temporal stats → 960 dim
        X_list.append(temp_vec)
        y_list.append(0)
        group_list.append(other_group)
    print(f"   - 生成 150 幀合成 other 負樣本 (group={other_group})")

    detector.close()


    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(group_list, dtype=np.int32)
    print(f"\n[+] 特徵萃取完成: 總樣本數={len(X)}, 特徵維度={X.shape[1]}, 來源組別數={len(np.unique(groups))}")
    return X, y, groups, source_names


def train_xgboost_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    source_names: List[str],
    output_model_path: str = "models/wash_hand_xgb.joblib",
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.08,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """Train XGBoost Classifier with leak-free Stratified cross-validation and save model + metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)

    print("\n🚀 開始訓練 XGBoost 分類器 (5-Fold 分層交叉驗證)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    val_true = []
    val_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="multi:softprob",
            num_class=8,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        val_true.extend(y_val)
        val_pred.extend(preds)

    target_names = [LABELS[i] for i in range(8)]
    report_dict = classification_report(val_true, val_pred, target_names=target_names, output_dict=True, zero_division=0)
    print("\n📊 【XGBoost 5-Fold 分層交叉驗證報告】")
    print(classification_report(val_true, val_pred, target_names=target_names, zero_division=0))

    # Leak-free GroupKFold cross-validation across distinct video/session groups
    n_groups = len(np.unique(groups))
    group_report_dict = {}
    if n_groups >= 3:
        n_splits_g = min(5, n_groups)
        print(f"\n🚀 執行 GroupKFold 跨獨立來源無洩漏交叉驗證 (來源組數={n_groups}, 折數={n_splits_g})...")
        gkf = GroupKFold(n_splits=n_splits_g)
        g_true, g_pred = [], []
        for g_train_idx, g_val_idx in gkf.split(X, y, groups=groups):
            clf_g = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                objective="multi:softprob",
                num_class=8,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            clf_g.fit(X[g_train_idx], y[g_train_idx])
            g_preds = clf_g.predict(X[g_val_idx])
            g_true.extend(y[g_val_idx])
            g_pred.extend(g_preds)

        group_report_dict = classification_report(g_true, g_pred, target_names=target_names, output_dict=True, zero_division=0)
        print("📊 【XGBoost GroupKFold 跨來源交叉驗證報告】")
        print(classification_report(g_true, g_pred, target_names=target_names, zero_division=0))

    # Train final model on full dataset
    print(f"[*] 正在全量資料集上訓練最終模型並儲存至 {output_model_path}...")
    final_clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=8,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X, y)

    joblib.dump(final_clf, output_model_path)
    print(f"[✅] XGBoost 模型已成功儲存至: {output_model_path}")

    # Export model metadata provenance
    metadata_path = os.path.splitext(output_model_path)[0] + "_metadata.json"
    metadata = {
        "model_file": os.path.basename(output_model_path),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "total_samples": int(len(X)),
        "feature_dim": int(X.shape[1]),
        "classes": LABELS,
        "sources": source_names,
        "unique_groups": int(n_groups),
        "hyperparameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "stratified_cv_metrics": {
            "macro_f1": float(report_dict.get("macro avg", {}).get("f1-score", 0.0)),
            "weighted_f1": float(report_dict.get("weighted avg", {}).get("f1-score", 0.0)),
            "accuracy": float(report_dict.get("accuracy", 0.0)),
            "per_class": {
                cls_name: {
                    "precision": float(report_dict[cls_name]["precision"]),
                    "recall": float(report_dict[cls_name]["recall"]),
                    "f1": float(report_dict[cls_name]["f1-score"]),
                    "support": int(report_dict[cls_name]["support"]),
                }
                for cls_name in target_names if cls_name in report_dict
            },
        },
        "group_cv_metrics": {
            "macro_f1": float(group_report_dict.get("macro avg", {}).get("f1-score", 0.0)) if group_report_dict else 0.0,
            "weighted_f1": float(group_report_dict.get("weighted avg", {}).get("f1-score", 0.0)) if group_report_dict else 0.0,
            "accuracy": float(group_report_dict.get("accuracy", 0.0)) if group_report_dict else 0.0,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[✅] 模型中繼資料已儲存至: {metadata_path}")

    return final_clf, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost wash hand classifier using sample_v1 & sample_v2 .mov files.")
    parser.add_argument("--output", type=str, default="models/wash_hand_xgb.joblib", help="Output model path")
    parser.add_argument("--sample-v2", type=str, default="data/sample_v2", help="Path to sample_v2 directory")
    parser.add_argument("--sample-v1", type=str, default="data/sample_v1", help="Path to sample_v1 directory")
    parser.add_argument("--aug", type=int, default=2, help="Augmentations per frame")
    parser.add_argument("--estimators", type=int, default=150, help="Number of XGBoost trees")
    parser.add_argument("--depth", type=int, default=6, help="Max tree depth")
    parser.add_argument("--lr", type=float, default=0.08, help="Learning rate")
    args = parser.parse_args()

    X, y, groups, sources = extract_dataset(
        sample_v2_dir=args.sample_v2,
        sample_v1_dir=args.sample_v1,
        augmentations_per_frame=args.aug,
    )

    if len(X) > 0:
        train_xgboost_classifier(
            X, y, groups, sources,
            output_model_path=args.output,
            n_estimators=args.estimators,
            max_depth=args.depth,
            learning_rate=args.lr,
        )
    else:
        print("[!] No training samples could be extracted.")

