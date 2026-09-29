"""
LSTM / GRU Temporal Model Training & Evaluation Module
Trains a sliding-window sequence classifier on 30-frame hand wash gesture features,
using strict person-based data splitting to prevent data leakage.
"""

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, List, Dict

from src.rule_classifier import LABELS, NAME_TO_LABEL, LABEL_SHORT_ZH


def build_lstm_model(seq_len: int = 30, feature_dim: int = 160, num_classes: int = 8) -> tf.keras.Model:
    """Build the LSTM model architecture defined in IMPLEMENTATION.md Phase 12."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, feature_dim)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_dataset_from_disk(data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load .npy sequences from data/processed/<person_id>/<action>_*.npy.
    Returns:
        X: array of shape (N, 30, feature_dim)
        y: array of shape (N,)
        groups: array of person_ids of shape (N,)
    """
    X_list = []
    y_list = []
    groups_list = []

    person_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    for pdir in person_dirs:
        if not os.path.isdir(pdir):
            continue
        person_id = os.path.basename(pdir)
        npy_files = glob.glob(os.path.join(pdir, "*.npy"))
        for npy_path in npy_files:
            fname = os.path.basename(npy_path)
            action_name = fname.split("_")[0]
            if action_name not in NAME_TO_LABEL:
                continue

            data = np.load(npy_path)
            if data.ndim == 2 and data.shape[0] == 30:
                X_list.append(data)
                y_list.append(NAME_TO_LABEL[action_name])
                groups_list.append(person_id)

    if not X_list:
        return np.empty((0, 30, 160)), np.empty((0,)), np.empty((0,))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(groups_list)


def generate_synthetic_dataset(
    num_persons: int = 10,
    clips_per_action: int = 15,
    seq_len: int = 30,
    feature_dim: int = 160,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic synthetic landmark sequences for demonstration & pipeline testing."""
    X_list = []
    y_list = []
    groups_list = []

    for p in range(1, num_persons + 1):
        person_id = f"person_{p:02d}"
        for label_idx, action_name in LABELS.items():
            for _ in range(clips_per_action):
                base_pattern = np.zeros((seq_len, feature_dim), dtype=np.float32)
                # Distinct signature per action
                t = np.linspace(0, 2 * np.pi, seq_len)[:, None]
                freq = (label_idx + 1) * 0.5
                base_pattern[:, :20] = np.sin(freq * t) * 0.5 + (label_idx * 0.2)
                base_pattern[:, 20:40] = np.cos(freq * t) * 0.5
                # Add gaussian noise
                noise = np.random.normal(0, 0.05, base_pattern.shape).astype(np.float32)
                sample = base_pattern + noise

                X_list.append(sample)
                y_list.append(label_idx)
                groups_list.append(person_id)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32), np.array(groups_list)


def train_and_evaluate(
    data_dir: str = "data/processed",
    output_model_path: str = "models/wash_hand_lstm.keras",
    epochs: int = 30,
    batch_size: int = 16,
    use_synthetic_if_empty: bool = True,
):
    print("[*] Loading dataset...")
    X, y, groups = load_dataset_from_disk(data_dir)

    if len(X) == 0:
        if use_synthetic_if_empty:
            print("[!] No recorded data found in data/processed/. Generating synthetic dataset for demonstration...")
            X, y, groups = generate_synthetic_dataset(num_persons=10, clips_per_action=12)
        else:
            print("[Error] No dataset found and synthetic mode disabled.")
            return

    unique_persons = np.unique(groups)
    print(f"[+] Loaded {len(X)} samples across {len(unique_persons)} participants: {list(unique_persons)}")

    # Split persons: 70% train, 15% val, 15% test
    n_total = len(unique_persons)
    n_train = max(1, int(n_total * 0.7))
    n_val = max(1, int(n_total * 0.15))

    train_persons = unique_persons[:n_train]
    val_persons = unique_persons[n_train : n_train + n_val]
    test_persons = unique_persons[n_train + n_val :]
    if len(test_persons) == 0:
        test_persons = val_persons

    print(f"    Train participants: {list(train_persons)}")
    print(f"    Val participants:   {list(val_persons)}")
    print(f"    Test participants:  {list(test_persons)}")

    train_idx = np.isin(groups, train_persons)
    val_idx = np.isin(groups, val_persons)
    test_idx = np.isin(groups, test_persons)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    feature_dim = X.shape[2]
    model = build_lstm_model(seq_len=30, feature_dim=feature_dim, num_classes=len(LABELS))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
    ]

    print(f"\n[*] Training LSTM Model for {epochs} epochs...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[*] Evaluating on Test Set (Unseen Participants)...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    target_names = [f"{LABELS[i]} ({LABEL_SHORT_ZH[LABELS[i]]})" for i in range(len(LABELS))]
    print("\n================ Classification Report ================")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    print("\n================ Confusion Matrix ================")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.save(output_model_path)
    print(f"\n[+] Successfully saved trained model to: {output_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wash Hand Gesture LSTM Model")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to processed .npy data")
    parser.add_argument("--output", type=str, default="models/wash_hand_lstm.keras", help="Model save path")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data for testing")
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic_if_empty=True,
    )
