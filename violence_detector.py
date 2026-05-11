"""
Violence Detection Pipeline
============================
Stage 1 : YOLOv8-nano  — person detection & tracking
Stage 2 : CNN+LSTM     — per-person violence classification
Stage 3 : Fusion       — frame-level verdict from all tracked persons

Usage:
    python violence_detector.py --input <video_path> --output <out_path> [--weights <model.h5>]

If --weights is omitted the model is built and left un-trained (random weights).
Pass a trained .h5 / .keras file to get meaningful predictions.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── TF / Keras ────────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    TimeDistributed,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
)
from tensorflow.keras.optimizers import Adam

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════
SEQUENCE_LENGTH = 20      # frames per person sequence
IMG_H, IMG_W   = 64, 64  # crop dimensions fed to CNN
YOLO_CONF      = 0.5     # minimum YOLO confidence
VIOLENCE_THRESH = 0.5    # score above which → VIOLENT


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════

def create_cnn_lstm() -> Sequential:
    """
    Build and compile the CNN+LSTM violence-classification model.

    Input  : (batch, 20, 64, 64, 3)
    Output : (batch, 1)  — sigmoid probability (1 = violent)
    """
    model = Sequential(name="CNN_LSTM_Violence_Classifier")

    # ── CNN Block 1 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same"),
        input_shape=(SEQUENCE_LENGTH, IMG_H, IMG_W, 3),
        name="td_conv1",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)),  name="td_pool1"))
    model.add(TimeDistributed(BatchNormalization(),          name="td_bn1"))
    model.add(TimeDistributed(Dropout(0.4),                 name="td_drop1"))

    # ── CNN Block 2 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv2",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)),  name="td_pool2"))
    model.add(TimeDistributed(BatchNormalization(),          name="td_bn2"))
    model.add(TimeDistributed(Dropout(0.4),                 name="td_drop2"))

    # ── CNN Block 3 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(128, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv3",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)),  name="td_pool3"))
    model.add(TimeDistributed(BatchNormalization(),          name="td_bn3"))
    model.add(TimeDistributed(Dropout(0.4),                 name="td_drop3"))

    # ── CNN Block 4 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv4",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)),  name="td_pool4"))
    model.add(TimeDistributed(BatchNormalization(),          name="td_bn4"))
    model.add(TimeDistributed(Dropout(0.4),                 name="td_drop4"))

    # ── Global Average Pooling (NOT Flatten) ─────────────────────────────────
    model.add(TimeDistributed(GlobalAveragePooling2D(),      name="td_gap"))

    # ── Stacked LSTM (3 layers) ───────────────────────────────────────────────
    model.add(LSTM(256, return_sequences=True,  dropout=0.4, recurrent_dropout=0.3, name="lstm_1"))
    model.add(LSTM(256, return_sequences=True,  dropout=0.4, recurrent_dropout=0.3, name="lstm_2"))
    model.add(LSTM(256, return_sequences=False, dropout=0.4, recurrent_dropout=0.3, name="lstm_3"))

    # ── Output ───────────────────────────────────────────────────────────────
    model.add(Dense(1, activation="sigmoid", name="output"))

    # ── Compile ──────────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _color(prob: float):
    """Return (B,G,R) tuple — red if violent, green if not."""
    return (0, 0, 255) if prob >= VIOLENCE_THRESH else (0, 255, 0)


def draw_person_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    prob: float,
) -> None:
    """Draw bounding box + label for a single tracked person."""
    color = _color(prob)
    label = f"ID:{track_id}  {prob:.2f}"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    # Label background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y1 - 6, th + baseline)
    cv2.rectangle(frame, (x1, label_y - th - baseline), (x1 + tw, label_y), color, cv2.FILLED)
    cv2.putText(
        frame, label,
        (x1, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_verdict(frame: np.ndarray, final_score: float, label: str) -> None:
    """Draw the frame-level verdict in the top-left corner."""
    color = _color(final_score)
    text  = f"{label}  score:{final_score:.2f}"

    # Semi-transparent background bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (360, 40), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame, text,
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — YOLO PERSON EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_person_crops(
    frame: np.ndarray,
    results,
    sequence_buffer: defaultdict,
) -> dict:
    """
    Parse YOLO tracking results for a single frame.

    For every detected person:
      - Clamp bbox to frame boundaries
      - Crop → resize (64,64) → normalize [0,1]
      - Append to the person's sequence buffer (keep last 20)

    Returns
    -------
    track_boxes : dict  {track_id: (x1, y1, x2, y2)}
        Bounding boxes for all persons detected in this frame.
    """
    h, w = frame.shape[:2]
    track_boxes: dict = {}

    if results is None or len(results) == 0:
        return track_boxes

    result = results[0]  # single-frame result

    # results.boxes can be None when no detections
    if result.boxes is None or result.boxes.id is None:
        return track_boxes

    boxes   = result.boxes.xyxy.cpu().numpy()    # (N, 4)
    ids     = result.boxes.id.cpu().numpy()      # (N,)
    confs   = result.boxes.conf.cpu().numpy()    # (N,)

    for box, track_id, conf in zip(boxes, ids, confs):
        if conf < YOLO_CONF:
            continue

        track_id = int(track_id)

        # ── Clamp coordinates ────────────────────────────────────────────────
        x1 = int(max(0, box[0]))
        y1 = int(max(0, box[1]))
        x2 = int(min(w - 1, box[2]))
        y2 = int(min(h - 1, box[3]))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        track_boxes[track_id] = (x1, y1, x2, y2)

        # ── Crop → resize → normalize ────────────────────────────────────────
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (IMG_W, IMG_H))          # (64, 64, 3)
        crop = crop.astype(np.float32) / 255.0           # [0, 1]

        # ── Append to buffer, keep last SEQUENCE_LENGTH frames ───────────────
        sequence_buffer[track_id].append(crop)
        if len(sequence_buffer[track_id]) > SEQUENCE_LENGTH:
            sequence_buffer[track_id] = sequence_buffer[track_id][-SEQUENCE_LENGTH:]

    return track_boxes


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2+3 — CNN+LSTM INFERENCE + FUSION
# ═════════════════════════════════════════════════════════════════════════════

def classify_and_fuse(
    cnn_lstm_model,
    sequence_buffer: defaultdict,
    track_boxes: dict,
) -> tuple:
    """
    Run CNN+LSTM on every person whose buffer has exactly SEQUENCE_LENGTH frames.

    Returns
    -------
    person_probs : dict   {track_id: float}   — per-person violence probability
    final_score  : float  — average across all classified persons (0.0 if none)
    label        : str    — "VIOLENT" or "NON-VIOLENT"
    """
    person_probs: dict = {}

    for track_id, (x1, y1, x2, y2) in track_boxes.items():
        buf = sequence_buffer[track_id]

        # Only classify when the buffer is full (exactly 20 frames)
        if len(buf) != SEQUENCE_LENGTH:
            continue

        # Stack → (20, 64, 64, 3) → expand → (1, 20, 64, 64, 3)
        sequence = np.stack(buf, axis=0)                    # (20, 64, 64, 3)
        sequence = np.expand_dims(sequence, axis=0)         # (1, 20, 64, 64, 3)

        prob = float(cnn_lstm_model.predict(sequence, verbose=0)[0][0])
        person_probs[track_id] = prob

    # ── Fusion ───────────────────────────────────────────────────────────────
    if person_probs:
        final_score = float(np.mean(list(person_probs.values())))
    else:
        final_score = 0.0

    label = "VIOLENT" if final_score >= VIOLENCE_THRESH else "NON-VIOLENT"
    return person_probs, final_score, label


# ═════════════════════════════════════════════════════════════════════════════
# MAIN VIDEO PROCESSING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def process_video(
    input_path: str,
    output_path: str,
    yolo_model_path: str = "yolov8n.pt",
    cnn_lstm_weights: str | None = None,
) -> None:
    """
    Full end-to-end violence detection pipeline on a video file.

    Parameters
    ----------
    input_path        : path to source video
    output_path       : path to write annotated output video
    yolo_model_path   : YOLOv8 weights (default: yolov8n.pt)
    cnn_lstm_weights  : optional path to trained CNN+LSTM weights (.h5 / .keras)
    """

    # ── Load models ──────────────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 model …")
    yolo = YOLO(yolo_model_path)

    print("[INFO] Building CNN+LSTM model …")
    cnn_lstm = create_cnn_lstm()
    cnn_lstm.summary()

    if cnn_lstm_weights and Path(cnn_lstm_weights).is_file():
        print(f"[INFO] Loading CNN+LSTM weights from: {cnn_lstm_weights}")
        cnn_lstm.load_weights(cnn_lstm_weights)
    else:
        print("[WARN] No trained weights provided — predictions will be random.")

    # ── Open video source ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps  ({total} frames total)")

    # ── VideoWriter ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Per-track sequence buffer ─────────────────────────────────────────────
    sequence_buffer: defaultdict = defaultdict(list)

    frame_idx = 0

    print("[INFO] Starting frame-by-frame inference …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO]  … processed {frame_idx}/{total} frames")

        # ── STAGE 1: YOLO tracking ────────────────────────────────────────────
        results = yolo.track(
            frame,
            persist=True,
            classes=[0],        # persons only
            conf=YOLO_CONF,
            verbose=False,
        )

        track_boxes = extract_person_crops(frame, results, sequence_buffer)

        # ── STAGE 2+3: CNN+LSTM + Fusion ──────────────────────────────────────
        if track_boxes:
            person_probs, final_score, verdict = classify_and_fuse(
                cnn_lstm, sequence_buffer, track_boxes
            )
        else:
            # No persons detected → skip classifier for this frame
            person_probs = {}
            final_score  = 0.0
            verdict      = "NON-VIOLENT"

        # ── VISUALIZATION ─────────────────────────────────────────────────────
        # Per-person bounding boxes
        for track_id, (x1, y1, x2, y2) in track_boxes.items():
            prob = person_probs.get(track_id, 0.0)
            draw_person_box(frame, x1, y1, x2, y2, track_id, prob)

        # Frame-level verdict
        draw_verdict(frame, final_score, verdict)

        writer.write(frame)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    print(f"[INFO] Done. Annotated video saved to: {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-stage violence detection: YOLO + CNN+LSTM"
    )
    parser.add_argument(
        "--input",  "-i",
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output_violence_detected.mp4",
        help="Path to save the annotated output video (default: output_violence_detected.mp4).",
    )
    parser.add_argument(
        "--yolo",
        default="yolov8n.pt",
        help="YOLOv8 model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--weights", "-w",
        default=None,
        help="Path to trained CNN+LSTM weights (.h5 or .keras). "
             "Omit to run with random weights (for architecture testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        yolo_model_path=args.yolo,
        cnn_lstm_weights=args.weights,
    )
