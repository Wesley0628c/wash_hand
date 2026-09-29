"""
Temporal Feature Aggregator & Landmark Dropout Augmentation Module
Extracts statistical aggregations over sliding windows (15 frames)
and provides Landmark Dropout to simulate heavy foam occlusion and hand overlap.
"""

from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.features import extract_hand_features, feature_dict_to_vector


def apply_landmark_dropout(
    left_hand: Optional[np.ndarray],
    right_hand: Optional[np.ndarray],
    point_dropout_prob: float = 0.20,
    single_hand_drop_prob: float = 0.15,
    jitter_std: float = 0.015,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Augments landmarks by:
    1. Randomly dropping (zeroing out) a subset of landmark points to simulate soap foam.
    2. Randomly dropping one hand completely (simulating severe occlusion).
    3. Adding small coordinate jitter.
    """
    aug_left = left_hand.copy() if left_hand is not None else None
    aug_right = right_hand.copy() if right_hand is not None else None

    # 1. Single Hand Drop
    if aug_left is not None and aug_right is not None and np.random.rand() < single_hand_drop_prob:
        if np.random.rand() < 0.5:
            aug_left = None
        else:
            aug_right = None

    # 2. Point Landmark Dropout
    if aug_left is not None:
        mask = np.random.rand(21) < point_dropout_prob
        aug_left[mask] = 0.0
        # Add Jitter
        noise = np.random.normal(0, jitter_std, aug_left.shape).astype(np.float32)
        aug_left[~mask] += noise[~mask]

    if aug_right is not None:
        mask = np.random.rand(21) < point_dropout_prob
        aug_right[mask] = 0.0
        # Add Jitter
        noise = np.random.normal(0, jitter_std, aug_right.shape).astype(np.float32)
        aug_right[~mask] += noise[~mask]

    return aug_left, aug_right


class TemporalFeatureBuffer:
    """
    Maintains a rolling buffer of 160-dim frame feature vectors (default 15 frames)
    and computes rich statistical aggregations (mean, std, min, max, delta, current).
    """

    def __init__(self, buffer_size: int = 15, base_dim: int = 160):
        self.buffer_size = buffer_size
        self.base_dim = base_dim
        self.history = deque(maxlen=buffer_size)

    def reset(self):
        self.history.clear()

    def update(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Add new 160-dim vector and return flattened temporal summary vector.
        Output dimension: base_dim * 6 = 160 * 6 = 960
        """
        self.history.append(feature_vec)

        arr = np.array(self.history, dtype=np.float32)  # Shape: (K, 160)
        curr = feature_vec
        mean_val = np.mean(arr, axis=0)
        std_val = np.std(arr, axis=0) if len(arr) > 1 else np.zeros(self.base_dim, dtype=np.float32)
        min_val = np.min(arr, axis=0)
        max_val = np.max(arr, axis=0)
        delta_val = (arr[-1] - arr[0]) if len(arr) > 1 else np.zeros(self.base_dim, dtype=np.float32)

        return np.concatenate([curr, mean_val, std_val, min_val, max_val, delta_val]).astype(np.float32)
