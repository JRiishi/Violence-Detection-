"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         END-TO-END VIOLENCE DETECTION PIPELINE  (pipeline.py)              ║
║                                                                              ║
║  STAGE 0 ── Data extraction  (Violence/ & NonViolence/ folders)             ║
║  STAGE 1 ── YOLOv8-nano      (person detection + tracking)                 ║
║  STAGE 2 ── CNN+LSTM         (per-person violence classification)           ║
║  STAGE 3 ── Fusion           (frame-level verdict)                          ║
║                                                                              ║
║  Run modes (pick one):                                                       ║
║    # Full pipeline: extract data → train → infer                            ║
║    python pipeline.py --mode full   --input test.mp4                        ║
║                                                                              ║
║    # Train only (skip inference)                                             ║
║    python pipeline.py --mode train                                           ║
║                                                                              ║
║    # Infer only (skip training, load existing weights)                      ║
║    python pipeline.py --mode infer  --input test.mp4 --weights model.h5    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

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
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 20          # frames per person sequence
IMG_H, IMG_W     = 64, 64      # crop size fed to the CNN
YOLO_CONF        = 0.5         # minimum YOLO detection confidence
VIOLENCE_THRESH  = 0.5         # probability threshold → VIOLENT
VIDEO_EXTS       = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def create_cnn_lstm() -> Sequential:
    """
    Build and compile the CNN+LSTM violence-classification model.

    Input  shape : (batch, 20, 64, 64, 3)
    Output shape : (batch, 1)  — sigmoid probability  (1 = violent)

    Architecture
    ────────────
    4 × CNN blocks (TimeDistributed):
        Conv2D → MaxPool2D → BatchNorm → Dropout
    TimeDistributed(GlobalAveragePooling2D)   ← NOT Flatten
    3 stacked LSTMs  (256 units each, first two return_sequences=True)
    Dense(1, sigmoid)

    Compiled with Adam(lr=1e-3) + binary_crossentropy
    """
    model = Sequential(name="CNN_LSTM_Violence_Classifier")

    # ── CNN Block 1 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(32, kernel_size=(5, 5), activation="relu", padding="same"),
        input_shape=(SEQUENCE_LENGTH, IMG_H, IMG_W, 3),
        name="td_conv1",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)), name="td_pool1"))
    model.add(TimeDistributed(BatchNormalization(),         name="td_bn1"))
    model.add(TimeDistributed(Dropout(0.4),                name="td_drop1"))

    # ── CNN Block 2 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(64, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv2",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)), name="td_pool2"))
    model.add(TimeDistributed(BatchNormalization(),         name="td_bn2"))
    model.add(TimeDistributed(Dropout(0.4),                name="td_drop2"))

    # ── CNN Block 3 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(128, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv3",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)), name="td_pool3"))
    model.add(TimeDistributed(BatchNormalization(),         name="td_bn3"))
    model.add(TimeDistributed(Dropout(0.4),                name="td_drop3"))

    # ── CNN Block 4 ──────────────────────────────────────────────────────────
    model.add(TimeDistributed(
        Conv2D(256, kernel_size=(5, 5), activation="relu", padding="same"),
        name="td_conv4",
    ))
    model.add(TimeDistributed(MaxPool2D(pool_size=(3, 3)), name="td_pool4"))
    model.add(TimeDistributed(BatchNormalization(),         name="td_bn4"))
    model.add(TimeDistributed(Dropout(0.4),                name="td_drop4"))

    # ── Global Average Pooling  (NOT Flatten) ────────────────────────────────
    model.add(TimeDistributed(GlobalAveragePooling2D(), name="td_gap"))

    # ── Stacked LSTMs ────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 0 — DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_sequences_from_video(video_path: str, max_sequences: int = 3, yolo_model=None) -> list:
    """
    Read every frame of a video, resize to (IMG_H, IMG_W), normalise to [0,1],
    then slice into non-overlapping windows of SEQUENCE_LENGTH frames.
    If yolo_model is provided, tracking logic is used to extract human crops.

    Returns a list of np.ndarray of shape (SEQUENCE_LENGTH, IMG_H, IMG_W, 3).
    Returns [] if the video is too short or cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {Path(video_path).name}")
        return []

    sequences = []
    
    if yolo_model is not None:
        sequence_buffer = defaultdict(list)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = yolo_model.track(
                frame, persist=True, classes=[0], conf=YOLO_CONF, verbose=False
            )
            _extract_person_crops(frame, results, sequence_buffer)
            
            for tid in list(sequence_buffer.keys()):
                if len(sequence_buffer[tid]) == SEQUENCE_LENGTH:
                    seq = np.stack(sequence_buffer[tid], axis=0)
                    sequences.append(seq)
                    sequence_buffer[tid] = []
                    
                    if len(sequences) >= max_sequences:
                        break
            if len(sequences) >= max_sequences:
                break
        cap.release()
        return sequences

    # Fallback to pure frames if yolo_model is None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_W, IMG_H))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    if len(frames) < SEQUENCE_LENGTH:
        print(f"  [SKIP] Too short ({len(frames)} frames): {Path(video_path).name}")
        return []

    num_possible = len(frames) // SEQUENCE_LENGTH
    num_to_take  = min(num_possible, max_sequences)

    indices   = np.linspace(0, num_possible - 1, num_to_take, dtype=int)
    for idx in indices:
        start = idx * SEQUENCE_LENGTH
        seq   = np.stack(frames[start : start + SEQUENCE_LENGTH], axis=0)
        sequences.append(seq)

    return sequences


def load_dataset(data_dir: str, max_seqs_per_video: int = 3, yolo_model=None) -> tuple:
    """
    Walk Violence/ and NonViolence/ sub-folders under `data_dir`.

    Folder name rules (case-insensitive, ignores _ and -):
        starts with "nonviolence"  → label 0
        starts with "violence"     → label 1

    Returns
    ───────
    X : np.ndarray  shape (N, 20, 64, 64, 3)   float32  [0, 1]
    y : np.ndarray  shape (N,)                  float32  {0, 1}
    """
    data_dir = Path(data_dir)
    folder_label_map: dict[Path, int] = {}

    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower().replace("_", "").replace("-", "")
        if name.startswith("nonviolence"):
            folder_label_map[folder] = 0
        elif name.startswith("violence"):
            folder_label_map[folder] = 1

    if not folder_label_map:
        raise RuntimeError(
            f"No Violence/NonViolence folders found under: {data_dir}\n"
            "Expected names like: Violence, NonViolence, Violence_2, NonViolence_2"
        )

    print(f"\n{'─'*60}")
    print(f"  STAGE 0 — DATA EXTRACTION")
    print(f"{'─'*60}")
    print(f"  Root : {data_dir}")
    print(f"  Found {len(folder_label_map)} folder(s):\n")
    for f, lbl in folder_label_map.items():
        print(f"    label={lbl}  ←  {f.name}/")

    all_seqs:   list = []
    all_labels: list = []

    for folder, label in folder_label_map.items():
        video_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        print(f"\n  [{folder.name}/]  {len(video_files)} videos  (label={label})")

        for i, vpath in enumerate(video_files, 1):
            seqs = _extract_sequences_from_video(str(vpath), max_sequences=max_seqs_per_video, yolo_model=yolo_model)
            for seq in seqs:
                all_seqs.append(seq)
                all_labels.append(float(label))

            if i % 100 == 0:
                print(f"    … {i}/{len(video_files)} done  |  seqs collected: {len(all_seqs)}")

    if not all_seqs:
        raise RuntimeError("No sequences extracted — check your video files.")

    X = np.stack(all_seqs, axis=0).astype(np.float32)   # (N, 20, 64, 64, 3)
    y = np.array(all_labels, dtype=np.float32)           # (N,)

    print(f"\n  ✔  Total sequences : {X.shape[0]}")
    print(f"     Violent         : {int(y.sum())}")
    print(f"     Non-violent     : {int((1 - y).sum())}")
    print(f"     X shape         : {X.shape}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_training(
    data_dir: str           = ".",
    epochs: int             = 30,
    batch_size: int         = 4,
    val_split: float        = 0.15,
    weights_out: str        = "model_weights.weights.h5",
    max_seqs_per_video: int = 3,
    seed: int               = 42,
    yolo_model_path: str    = "yolov8n.pt",
) -> str:
    """
    Full training pipeline.

    Steps
    ─────
    1. Extract sequences from Violence/ & NonViolence/ folders
    2. Stratified train/val split
    3. Build CNN+LSTM model
    4. Train with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    5. Save best weights → `weights_out`

    Returns path to saved weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ── Step 1: Extract data ──────────────────────────────────────────────────
    print(f"\n  Loading YOLO model for data extraction: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    X, y = load_dataset(data_dir, max_seqs_per_video=max_seqs_per_video, yolo_model=yolo_model)

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y,
    )

    print(f"\n{'─'*60}")
    print(f"  TRAINING  —  {len(X_train)} train  /  {len(X_val)} val sequences")
    print(f"{'─'*60}\n")

    # ── Step 3: Build model ───────────────────────────────────────────────────
    model = create_cnn_lstm()
    model.summary()

    # ── Step 4: Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=weights_out,
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

    # ── Step 5: Class-weight balancing ────────────────────────────────────────
    n_total   = len(y_train)
    n_violent = int(y_train.sum())
    n_non     = n_total - n_violent
    cw = {
        0: (1.0 / n_non)    * (n_total / 2.0) if n_non    > 0 else 1.0,
        1: (1.0 / n_violent) * (n_total / 2.0) if n_violent > 0 else 1.0,
    }
    print(f"  Class weights  →  non-violent: {cw[0]:.3f}   violent: {cw[1]:.3f}\n")

    # ── Step 6: Fit ───────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=callbacks,
        shuffle=True,
    )

    # ── Step 7: Final save + report ───────────────────────────────────────────
    model.save_weights(weights_out)

    best_acc  = max(history.history.get("val_accuracy",  [0.0]))
    best_loss = min(history.history.get("val_loss", [float("inf")]))
    print(f"\n{'─'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val_accuracy : {best_acc:.4f}")
    print(f"  Best val_loss     : {best_loss:.4f}")
    print(f"  Weights saved to  : {weights_out}")
    print(f"{'─'*60}\n")

    return weights_out


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — YOLO PERSON DETECTION & TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def _extract_person_crops(
    frame: np.ndarray,
    results,
    sequence_buffer: defaultdict,
) -> dict:
    """
    Parse one frame's YOLO results.

    For every tracked person:
      - Clamp bbox to frame bounds
      - Crop → resize (64×64) → normalise [0,1]
      - Append to sequence_buffer[track_id], keep last SEQUENCE_LENGTH frames

    Returns {track_id: (x1, y1, x2, y2)} for this frame.
    """
    h, w        = frame.shape[:2]
    track_boxes = {}

    if not results or len(results) == 0:
        return track_boxes

    result = results[0]
    if result.boxes is None or result.boxes.id is None:
        return track_boxes

    boxes = result.boxes.xyxy.cpu().numpy()   # (N, 4)
    ids   = result.boxes.id.cpu().numpy()     # (N,)
    confs = result.boxes.conf.cpu().numpy()   # (N,)

    for box, track_id, conf in zip(boxes, ids, confs):
        if conf < YOLO_CONF:
            continue

        track_id = int(track_id)

        # Clamp to frame boundaries
        x1 = int(max(0, box[0]))
        y1 = int(max(0, box[1]))
        x2 = int(min(w - 1, box[2]))
        y2 = int(min(h - 1, box[3]))

        if x2 <= x1 or y2 <= y1:
            continue   # degenerate box

        track_boxes[track_id] = (x1, y1, x2, y2)

        # Crop → resize → normalise
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (IMG_W, IMG_H)).astype(np.float32) / 255.0

        # Maintain rolling buffer (keep last SEQUENCE_LENGTH)
        sequence_buffer[track_id].append(crop)
        if len(sequence_buffer[track_id]) > SEQUENCE_LENGTH:
            sequence_buffer[track_id] = sequence_buffer[track_id][-SEQUENCE_LENGTH:]

    return track_boxes


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2+3 — CNN+LSTM INFERENCE + FUSION
# ══════════════════════════════════════════════════════════════════════════════

def _classify_and_fuse(
    cnn_lstm_model,
    sequence_buffer: defaultdict,
    track_boxes: dict,
) -> tuple:
    """
    Run CNN+LSTM on each tracked person (only when buffer == SEQUENCE_LENGTH).
    Fuse results → frame-level verdict.

    Returns
    ───────
    person_probs : {track_id: float}
    final_score  : float   (0.0 if nobody had enough frames)
    label        : "VIOLENT" | "NON-VIOLENT"
    """
    person_probs = {}

    for track_id in track_boxes:
        buf = sequence_buffer[track_id]
        if len(buf) != SEQUENCE_LENGTH:
            continue   # not enough frames yet

        seq  = np.stack(buf, axis=0)               # (20, 64, 64, 3)
        seq  = np.expand_dims(seq, axis=0)          # (1, 20, 64, 64, 3)
        prob = float(cnn_lstm_model.predict(seq, verbose=0)[0][0])
        person_probs[track_id] = prob

    # Frame-level fusion: average across all classified persons
    if person_probs:
        final_score = float(np.mean(list(person_probs.values())))
    else:
        final_score = 0.0

    label = "VIOLENT" if final_score >= VIOLENCE_THRESH else "NON-VIOLENT"
    return person_probs, final_score, label


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _bgr(prob: float):
    """RED if violent, GREEN if not (BGR tuples for OpenCV)."""
    return (0, 0, 255) if prob >= VIOLENCE_THRESH else (0, 255, 0)


def _draw_person_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    track_id: int,
    prob: float,
) -> None:
    color = _bgr(prob)
    label = f"ID:{track_id}  {prob:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ly = max(y1 - 6, th + bl)
    cv2.rectangle(frame, (x1, ly - th - bl), (x1 + tw, ly), color, cv2.FILLED)
    cv2.putText(frame, label, (x1, ly - bl),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_verdict(frame: np.ndarray, score: float, label: str) -> None:
    color = _bgr(score)
    text  = f"{label}  score:{score:.2f}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (370, 42), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, text, (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_inference(
    input_path: str,
    output_path: str,
    yolo_path: str  = "yolov8n.pt",
    weights: str | None = None,
) -> None:
    """
    Full inference pipeline on a video file.

    For every frame:
      1. YOLO tracks persons  →  person crops stored in rolling buffers
      2. When a buffer reaches 20 frames → CNN+LSTM predicts violence probability
      3. Frame-level verdict = mean of all persons' probabilities
      4. Annotated frame written to output video
    """
    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  STAGE 1 — Loading YOLOv8 model: {yolo_path}")
    yolo = YOLO(yolo_path)

    print(f"  STAGE 2 — Building CNN+LSTM model")
    cnn_lstm = create_cnn_lstm()
    cnn_lstm.summary()

    if weights and Path(weights).is_file():
        print(f"  Loading weights: {weights}")
        cnn_lstm.load_weights(weights)
    else:
        print("  [WARN] No weights loaded — predictions will be RANDOM.")
    print(f"{'─'*60}\n")

    # ── Open source video ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input  : {width}×{height} @ {fps:.1f} fps  ({total} frames)")
    print(f"  Output : {output_path}\n")

    # ── VideoWriter ───────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # ── Per-track sequence buffer (Stage 1 → Stage 2 handoff) ────────────────
    sequence_buffer: defaultdict = defaultdict(list)

    frame_idx = 0
    print("  Processing frames …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"    … frame {frame_idx}/{total}")

        # ── Stage 1 : YOLO person tracking ────────────────────────────────────
        results = yolo.track(
            frame,
            persist=True,       # maintain track IDs across frames
            classes=[0],        # persons only
            conf=YOLO_CONF,
            verbose=False,
        )

        track_boxes = _extract_person_crops(frame, results, sequence_buffer)

        # ── Stage 2+3 : CNN+LSTM + Fusion ─────────────────────────────────────
        if track_boxes:
            person_probs, final_score, verdict = _classify_and_fuse(
                cnn_lstm, sequence_buffer, track_boxes,
            )
        else:
            person_probs = {}
            final_score  = 0.0
            verdict      = "NON-VIOLENT"

        # ── Visualise ─────────────────────────────────────────────────────────
        for tid, (x1, y1, x2, y2) in track_boxes.items():
            _draw_person_box(frame, x1, y1, x2, y2, tid, person_probs.get(tid, 0.0))

        _draw_verdict(frame, final_score, verdict)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\n  ✔  Done — annotated video saved to: {output_path}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE  (train, then infer)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  FULL PIPELINE  :  EXTRACT → TRAIN → INFER")
    print("═" * 60)

    # Step A: Train
    weights_path = run_training(
        data_dir           = args.data_dir,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        val_split          = args.val_split,
        weights_out        = args.weights,
        max_seqs_per_video = args.max_seqs_per_video,
        seed               = args.seed,
        yolo_model_path    = args.yolo,
    )

    # Step B: Infer
    if not args.input:
        print("[INFO] --input not provided, skipping inference step.")
        return

    run_inference(
        input_path  = args.input,
        output_path = args.output,
        yolo_path   = args.yolo,
        weights     = weights_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end violence detection pipeline: extract → train → infer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode", choices=["full", "train", "infer"], default="full",
        help=(
            "full  = extract data → train → infer on --input video | "
            "train = data extraction + training only | "
            "infer = inference only (requires --weights)"
        ),
    )

    # Data / training
    parser.add_argument(
        "--data_dir", "-d", default=".",
        help="Root folder containing Violence/ and NonViolence/ sub-directories.",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=4,
        help="Training batch size. Reduce to 2 if you run out of RAM.",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.15,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument(
        "--max_seqs_per_video", type=int, default=3,
        help="Max non-overlapping 20-frame sequences to sample per video.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )

    # Inference
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input video path for inference (required for --mode infer / full).",
    )
    parser.add_argument(
        "--output", "-o", default="output_annotated.mp4",
        help="Path for the annotated output video.",
    )
    parser.add_argument(
        "--yolo", default="yolov8n.pt",
        help="YOLOv8 weights file.",
    )
    parser.add_argument(
        "--weights", "-w", default="model_weights.weights.h5",
        help=(
            "Path to save (training) or load (infer) CNN+LSTM weights. "
            "During training, best weights are saved here. "
            "During inference, weights are loaded from here."
        ),
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "full":
        if not args.input:
            sys.exit("[ERROR] --mode full requires --input <video_path>")
        run_full_pipeline(args)

    elif args.mode == "train":
        run_training(
            data_dir           = args.data_dir,
            epochs             = args.epochs,
            batch_size         = args.batch_size,
            val_split          = args.val_split,
            weights_out        = args.weights,
            max_seqs_per_video = args.max_seqs_per_video,
            seed               = args.seed,
            yolo_model_path    = args.yolo,
        )

    elif args.mode == "infer":
        if not args.input:
            sys.exit("[ERROR] --mode infer requires --input <video_path>")
        run_inference(
            input_path  = args.input,
            output_path = args.output,
            yolo_path   = args.yolo,
            weights     = args.weights,
        )
