"""
Training Script — Violence Detection CNN+LSTM
==============================================
Reads videos from:
    ./Violence/        (label = 1)
    ./Violence_2/      (label = 1)   ← used if present
    ./NonViolence/     (label = 0)
    ./NonViolence_2/   (label = 0)   ← used if present

For each video, extracts one or more non-overlapping sequences of
SEQUENCE_LENGTH (20) frames, resizes each frame to (64×64), and
normalises pixel values to [0, 1].

Usage:
    python train.py [--data_dir .] [--epochs 30] [--batch_size 4]
                    [--output model_weights.h5] [--val_split 0.15]
                    [--max_seqs_per_video 3]
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Import the model factory from the detector module
from violence_detector import create_cnn_lstm, SEQUENCE_LENGTH, IMG_H, IMG_W

# ─────────────────────────────────────────────────────────────────────────────
# Supported video extensions
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


# ═════════════════════════════════════════════════════════════════════════════
# DATA EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_sequences_from_video(
    video_path: str,
    max_sequences: int = 3,
) -> list:
    """
    Extract up to `max_sequences` non-overlapping sequences from a single video.

    Each sequence is a numpy array of shape (SEQUENCE_LENGTH, IMG_H, IMG_W, 3)
    with pixel values normalised to [0, 1].

    Returns a list of such arrays (may be empty if the video is too short).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return []

    # Collect all frames first (memory-efficient shortcut for small clips)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (IMG_W, IMG_H))
        frame_norm    = frame_resized.astype(np.float32) / 255.0
        frames.append(frame_norm)
    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        print(f"  [SKIP] Too short ({len(frames)} frames): {Path(video_path).name}")
        return []

    # Extract non-overlapping windows
    sequences = []
    num_possible = len(frames) // SEQUENCE_LENGTH
    num_to_take  = min(num_possible, max_sequences)

    # Evenly space the window start positions across the video
    indices = np.linspace(0, num_possible - 1, num_to_take, dtype=int)
    for idx in indices:
        start = idx * SEQUENCE_LENGTH
        seq   = np.stack(frames[start : start + SEQUENCE_LENGTH], axis=0)  # (20, 64, 64, 3)
        sequences.append(seq)

    return sequences


def load_dataset(
    data_dir: str,
    max_seqs_per_video: int = 3,
) -> tuple:
    """
    Walk the Violence / NonViolence folders under `data_dir` and build
    X (features) and y (labels) arrays.

    Folder names accepted (case-insensitive endings):
        *violence*   → label 1
        *nonviolence* / *non_violence* / *non-violence* → label 0

    Returns
    -------
    X : np.ndarray  shape (N, SEQUENCE_LENGTH, IMG_H, IMG_W, 3)
    y : np.ndarray  shape (N,)  — dtype float32, values 0 or 1
    """
    data_dir = Path(data_dir)

    # Map folder → label
    folder_label_map: dict[Path, int] = {}
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        name_lower = folder.name.lower().replace("_", "").replace("-", "")
        if name_lower.startswith("nonviolence"):
            folder_label_map[folder] = 0
        elif name_lower.startswith("violence"):
            folder_label_map[folder] = 1
        # ignore other directories (e.g. .venv, __pycache__)

    if not folder_label_map:
        raise RuntimeError(
            f"No Violence/NonViolence folders found under: {data_dir}\n"
            "Expected folder names like: Violence, NonViolence, Violence_2, NonViolence_2"
        )

    print(f"\n[INFO] Found {len(folder_label_map)} data folder(s):")
    for f, lbl in folder_label_map.items():
        print(f"  label={lbl}  ←  {f.name}/")

    all_sequences: list = []
    all_labels:    list = []

    for folder, label in folder_label_map.items():
        video_files = [
            p for p in sorted(folder.iterdir())
            if p.suffix.lower() in VIDEO_EXTS
        ]
        print(f"\n[INFO] Processing '{folder.name}/'  ({len(video_files)} videos, label={label}) …")

        for i, vpath in enumerate(video_files, 1):
            seqs = extract_sequences_from_video(str(vpath), max_sequences=max_seqs_per_video)
            for seq in seqs:
                all_sequences.append(seq)
                all_labels.append(float(label))

            if i % 50 == 0:
                print(f"  … {i}/{len(video_files)} videos done  (total seqs so far: {len(all_sequences)})")

    if not all_sequences:
        raise RuntimeError("No sequences extracted — check that your video files are valid.")

    X = np.stack(all_sequences, axis=0).astype(np.float32)  # (N, 20, 64, 64, 3)
    y = np.array(all_labels, dtype=np.float32)               # (N,)

    print(f"\n[INFO] Dataset ready: {X.shape[0]} sequences total")
    print(f"  Violent sequences    : {int(y.sum())}")
    print(f"  Non-violent sequences: {int((1 - y).sum())}")
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}\n")
    return X, y


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train(
    data_dir: str       = ".",
    epochs: int         = 30,
    batch_size: int     = 4,
    val_split: float    = 0.15,
    output_path: str    = "model_weights.h5",
    max_seqs_per_video: int = 3,
    seed: int           = 42,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y = load_dataset(data_dir, max_seqs_per_video=max_seqs_per_video)

    # ── 2. Train / validation split (stratified) ──────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=seed,
        stratify=y,
    )
    print(f"[INFO] Train: {len(X_train)} sequences  |  Val: {len(X_val)} sequences")

    # ── 3. Build model ────────────────────────────────────────────────────────
    print("[INFO] Building CNN+LSTM model …")
    model = create_cnn_lstm()
    model.summary()

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=output_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 5. Class weights (handle imbalance) ───────────────────────────────────
    n_total    = len(y_train)
    n_violent  = int(y_train.sum())
    n_non      = n_total - n_violent
    weight_for_0 = (1 / n_non)    * (n_total / 2.0) if n_non    > 0 else 1.0
    weight_for_1 = (1 / n_violent) * (n_total / 2.0) if n_violent > 0 else 1.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"[INFO] Class weights: non-violent={weight_for_0:.3f}, violent={weight_for_1:.3f}")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("[INFO] Starting training …\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=True,
    )

    # ── 7. Final save (in case best checkpoint was skipped) ───────────────────
    model.save_weights(output_path)
    print(f"\n[INFO] Training complete. Weights saved to: {output_path}")

    # ── 8. Quick summary ──────────────────────────────────────────────────────
    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    best_val_loss = min(history.history.get("val_loss", [float("inf")]))
    print(f"[INFO] Best val_accuracy : {best_val_acc:.4f}")
    print(f"[INFO] Best val_loss     : {best_val_loss:.4f}")
    print(
        f"\n[NEXT] Run inference with:\n"
        f"  python violence_detector.py --input <video.mp4> --weights {output_path}\n"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN+LSTM violence classifier on local video folders."
    )
    parser.add_argument(
        "--data_dir", "-d",
        default=".",
        help="Root directory containing Violence/ and NonViolence/ sub-folders "
             "(default: current directory).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int, default=4,
        help="Training batch size (default: 4). Lower if you run out of RAM/VRAM.",
    )
    parser.add_argument(
        "--val_split",
        type=float, default=0.15,
        help="Fraction of data used for validation (default: 0.15).",
    )
    parser.add_argument(
        "--output", "-o",
        default="model_weights.h5",
        help="Filename for the saved model weights (default: model_weights.h5).",
    )
    parser.add_argument(
        "--max_seqs_per_video",
        type=int, default=3,
        help="Max non-overlapping sequences to extract per video (default: 3). "
             "Set to 1 to speed up extraction at the cost of fewer training samples.",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_split=args.val_split,
        output_path=args.output,
        max_seqs_per_video=args.max_seqs_per_video,
        seed=args.seed,
    )
