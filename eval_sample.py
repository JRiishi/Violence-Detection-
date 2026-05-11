"""
eval_sample.py
══════════════
Picks 10 random videos from Violence/ and 10 from NonViolence/,
runs YOLO person-tracking + CNN+LSTM inference on each, and prints results.

Usage:
    python eval_sample.py
    python eval_sample.py --violence Violence --nonviolence NonViolence --weights model_weights.weights.h5 --yolo yolov8n.pt --n 10
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPool2D, BatchNormalization,
    Dropout, GlobalAveragePooling2D, LSTM, Dense,
)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match what the model was trained with)
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 20
IMG_H, IMG_W     = 64, 64
YOLO_CONF        = 0.4          # slightly lower for evaluation to catch more persons
VIOLENCE_THRESH  = 0.5
VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


# ─────────────────────────────────────────────────────────────────────────────
# Build the same CNN+LSTM architecture used during training
# ─────────────────────────────────────────────────────────────────────────────
def create_cnn_lstm() -> Sequential:
    model = Sequential(name="CNN_LSTM_Violence_Classifier")

    # Block 1 — (5,5) kernel, (2,2) pool
    model.add(TimeDistributed(
        Conv2D(32, (5, 5), activation="relu", padding="same"),
        input_shape=(SEQUENCE_LENGTH, IMG_H, IMG_W, 3), name="td_conv1"))
    model.add(TimeDistributed(MaxPool2D((2, 2)), name="td_pool1"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn1"))
    model.add(TimeDistributed(Dropout(0.4), name="td_drop1"))

    # Block 2 — (5,5) kernel, (2,2) pool
    model.add(TimeDistributed(Conv2D(64,  (5, 5), activation="relu", padding="same"), name="td_conv2"))
    model.add(TimeDistributed(MaxPool2D((2, 2)), name="td_pool2"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn2"))
    model.add(TimeDistributed(Dropout(0.4), name="td_drop2"))

    # Block 3 — (3,3) kernel, (2,2) pool
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation="relu", padding="same"), name="td_conv3"))
    model.add(TimeDistributed(MaxPool2D((2, 2)), name="td_pool3"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn3"))
    model.add(TimeDistributed(Dropout(0.4), name="td_drop3"))

    # Block 4 — (3,3) kernel, (2,2) pool
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation="relu", padding="same"), name="td_conv4"))
    model.add(TimeDistributed(MaxPool2D((2, 2)), name="td_pool4"))
    model.add(TimeDistributed(BatchNormalization(), name="td_bn4"))
    model.add(TimeDistributed(Dropout(0.4), name="td_drop4"))

    model.add(TimeDistributed(GlobalAveragePooling2D(), name="td_gap"))

    # LSTMs: 128 → 64 → 64  (matching attempt_2.ipynb)
    model.add(LSTM(128, return_sequences=True,  dropout=0.4, recurrent_dropout=0.3, name="lstm_1"))
    model.add(LSTM(64,  return_sequences=True,  dropout=0.4, recurrent_dropout=0.3, name="lstm_2"))
    model.add(LSTM(64,  return_sequences=False, dropout=0.4, recurrent_dropout=0.3, name="lstm_3"))

    model.add(Dense(1, activation="sigmoid", name="output"))
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# YOLO crop extraction (mirrors pipeline.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
def extract_person_crops(frame: np.ndarray, results, sequence_buffer: defaultdict) -> dict:
    track_boxes = {}
    if not results or len(results) == 0:
        return track_boxes
    result = results[0]
    if result.boxes is None or result.boxes.id is None:
        return track_boxes

    h, w    = frame.shape[:2]
    boxes   = result.boxes.xyxy.cpu().numpy()
    ids     = result.boxes.id.cpu().numpy()
    confs   = result.boxes.conf.cpu().numpy()

    for box, t_id, conf in zip(boxes, ids, confs):
        if conf < YOLO_CONF:
            continue
        t_id = int(t_id)
        x1 = int(max(0, box[0]));  y1 = int(max(0, box[1]))
        x2 = int(min(w - 1, box[2])); y2 = int(min(h - 1, box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        track_boxes[t_id] = (x1, y1, x2, y2)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (IMG_W, IMG_H)).astype(np.float32) / 255.0
        sequence_buffer[t_id].append(crop)
        if len(sequence_buffer[t_id]) > SEQUENCE_LENGTH:
            sequence_buffer[t_id].pop(0)

    return track_boxes


# ─────────────────────────────────────────────────────────────────────────────
# Single-video inference (YOLO + CNN-LSTM)
# ─────────────────────────────────────────────────────────────────────────────
def infer_video(video_path: str, yolo_model, cnn_lstm_model) -> dict:
    """
    Run YOLO person tracking + CNN+LSTM on a single video.

    Returns a dict with:
        verdict      : "VIOLENT" | "NON-VIOLENT" | "NO PERSONS DETECTED"
        final_score  : float  (average violence probability across all person-windows)
        n_windows    : int    (number of 20-frame person sequences evaluated)
        n_frames     : int    (total frames processed)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"verdict": "ERROR (cannot open)", "final_score": -1.0, "n_windows": 0, "n_frames": 0}

    sequence_buffer: defaultdict = defaultdict(list)
    all_probs   = []
    n_frames    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frames += 1

        # Stage 1: YOLO person tracking
        results = yolo_model.track(
            frame, persist=True, classes=[0], conf=YOLO_CONF, verbose=False
        )
        track_boxes = extract_person_crops(frame, results, sequence_buffer)

        # Stage 2: CNN+LSTM — classify any person who now has a full buffer
        for t_id in track_boxes:
            buf = sequence_buffer[t_id]
            if len(buf) == SEQUENCE_LENGTH:
                seq  = np.expand_dims(np.stack(buf, axis=0), axis=0)  # (1, 20, 64, 64, 3)
                prob = float(cnn_lstm_model.predict(seq, verbose=0)[0][0])
                all_probs.append(prob)

    cap.release()

    if not all_probs:
        return {"verdict": "NO PERSONS DETECTED", "final_score": 0.0, "n_windows": 0, "n_frames": n_frames}

    final_score = float(np.mean(all_probs))
    verdict     = "VIOLENT" if final_score >= VIOLENCE_THRESH else "NON-VIOLENT"
    return {"verdict": verdict, "final_score": final_score, "n_windows": len(all_probs), "n_frames": n_frames}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLO + CNN-LSTM evaluation on random video samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--violence",    "-v",  default="Violence",              help="Path to violence video folder")
    parser.add_argument("--nonviolence", "-nv", default="NonViolence",           help="Path to non-violence video folder")
    parser.add_argument("--weights",     "-w",  default="model_weights.weights.h5", help="CNN-LSTM weights file")
    parser.add_argument("--yolo",               default="yolov8n.pt",            help="YOLOv8 weights file")
    parser.add_argument("--n",           "-n",  type=int, default=10,            help="Number of videos to sample per class")
    parser.add_argument("--seed",               type=int, default=42,            help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Collect video paths ──────────────────────────────────────────────────
    def collect(folder: str, label: int):
        p = Path(folder)
        if not p.is_dir():
            sys.exit(f"[ERROR] Folder not found: {folder}")
        files = [f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        if len(files) < args.n:
            print(f"[WARN] Only {len(files)} videos in {folder}, sampling all of them.")
        return [(str(f), label) for f in random.sample(files, min(args.n, len(files)))]

    violence_samples    = collect(args.violence,    label=1)
    nonviolence_samples = collect(args.nonviolence, label=0)
    all_samples         = violence_samples + nonviolence_samples

    print(f"\n{'═'*70}")
    print(f"  YOLO + CNN-LSTM  —  Violence Detection Evaluation")
    print(f"{'═'*70}")
    print(f"  Violence folder   : {args.violence}  ({len(violence_samples)} videos sampled)")
    print(f"  NonViolence folder: {args.nonviolence}  ({len(nonviolence_samples)} videos sampled)")
    print(f"  Weights           : {args.weights}")
    print(f"  YOLO              : {args.yolo}")
    print(f"{'─'*70}\n")

    # ── Load models ──────────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 …")
    yolo = YOLO(args.yolo)

    print("[INFO] Building CNN+LSTM …")
    cnn_lstm = create_cnn_lstm()

    if Path(args.weights).is_file():
        print(f"[INFO] Loading weights from: {args.weights}\n")
        cnn_lstm.load_weights(args.weights)
    else:
        sys.exit(f"[ERROR] Weights file not found: {args.weights}")

    # ── Run inference on each sample ────────────────────────────────────────
    results_rows = []
    correct = 0

    print(f"{'─'*70}")
    header = f"{'#':>3}  {'File':<28}  {'TrueLabel':<11}  {'Verdict':<22}  {'Score':>6}  {'Windows':>7}  {'✓?':>3}"
    print(header)
    print(f"{'─'*70}")

    for i, (vpath, true_label) in enumerate(all_samples, 1):
        true_str = "VIOLENT" if true_label == 1 else "NON-VIOLENT"
        fname    = Path(vpath).name[:28]

        res = infer_video(vpath, yolo, cnn_lstm)

        is_correct = (
            (res["verdict"] == "VIOLENT"     and true_label == 1) or
            (res["verdict"] == "NON-VIOLENT" and true_label == 0)
        )
        if is_correct:
            correct += 1

        tick = "✓" if is_correct else "✗"
        row  = (
            f"{i:>3}  {fname:<28}  {true_str:<11}  "
            f"{res['verdict']:<22}  {res['final_score']:>6.3f}  "
            f"{res['n_windows']:>7}  {tick:>3}"
        )
        print(row)
        results_rows.append({
            "file": Path(vpath).name,
            "true_label": true_str,
            **res,
            "correct": is_correct,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    total  = len(all_samples)
    acc    = correct / total * 100 if total > 0 else 0.0
    vio_correct  = sum(1 for r in results_rows if r["true_label"] == "VIOLENT"     and r["correct"])
    nvio_correct = sum(1 for r in results_rows if r["true_label"] == "NON-VIOLENT" and r["correct"])

    print(f"\n{'═'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'─'*70}")
    print(f"  Total videos evaluated : {total}")
    print(f"  Overall accuracy       : {correct}/{total}  ({acc:.1f}%)")
    print(f"  Violence    correct    : {vio_correct}/{len(violence_samples)}")
    print(f"  NonViolence correct    : {nvio_correct}/{len(nonviolence_samples)}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
