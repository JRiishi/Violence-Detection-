# Violence Detection System - Project Architecture

## Directory Structure

```
Violence_Detector/
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore rules
│
├── Source Code
│   ├── violence_detector.py           # Core detection pipeline + model definition
│   ├── pipeline.py                    # End-to-end workflow (full/train/infer modes)
│   ├── train.py                       # Training script with sequence extraction
│   └── eval_sample.py                 # Evaluation on sample videos
│
├── Jupyter Notebooks
│   ├── main.ipynb                     # Primary training & visualization notebook
│   └── attempt_2.ipynb                # Experimental approaches & analysis
│
├── Models & Weights (in .gitignore)
│   ├── model_weights.h5               # Trained CNN+LSTM weights
│   └── yolov8n.pt                     # YOLOv8-nano pre-trained weights
│
└── Training Data (in .gitignore)
    ├── Violence/                      # Violent video samples (label=1)
    ├── Violence_2/                    # Additional violent samples
    ├── NonViolence/                   # Non-violent samples (label=0)
    └── NonViolence_2/                 # Additional non-violent samples
```

## System Architecture - Three-Stage Pipeline

### Stage 1: Person Detection & Tracking
**File**: `violence_detector.py`
- **Model**: YOLOv8-nano (`yolov8n.pt`)
- **Purpose**: Detect and track all persons in each video frame
- **Input**: Video frames
- **Output**: Bounding boxes with person IDs
- **Key Function**: `detect_persons_in_frame(frame, yolo_model, conf_threshold)`

### Stage 2: Violence Classification
**File**: `violence_detector.py`
- **Model**: CNN+LSTM
- **Architecture**:
  ```
  Input (20 frames, 64×64)
    ↓
  TimeDistributed(Conv2D + MaxPool + BatchNorm + Dropout) × 2
    ↓
  GlobalAveragePooling2D
    ↓
  LSTM (128 units, return_sequences=False)
    ↓
  Dense(64, relu) → Dropout(0.5)
    ↓
  Dense(32, relu) → Dropout(0.5)
    ↓
  Dense(1, sigmoid)  [Binary violence score]
  ```
- **Input**: 20-frame sequences (64×64 crop per person)
- **Output**: Violence probability [0, 1] per person
- **Key Function**: `create_cnn_lstm()`

### Stage 3: Fusion & Verdict
**File**: `violence_detector.py`
- **Purpose**: Aggregate per-person violence scores
- **Logic**: Frame marked VIOLENT if any person exceeds threshold (0.5)
- **Output**: Frame-level binary classification
- **Key Function**: `predict_video(video_path, model, yolo_model, ...)`

---

## Data Flow

### Training Pipeline
```
Raw Videos (Violence/, NonViolence/)
    ↓ [train.py]
Extract Non-overlapping Sequences (20 frames each)
    ↓
Resize to 64×64, Normalize [0, 1]
    ↓
Create Balanced Dataset
    ↓ [Train/Val Split]
Train CNN+LSTM Model
    ↓
Save Weights → model_weights.h5
```

### Inference Pipeline
```
Input Video
    ↓ [violence_detector.py]
Frame Extraction & YOLOv8 Detection
    ↓
For each tracked person:
    - Extract 64×64 crop
    - Build 20-frame sequence buffer
    - Feed to CNN+LSTM model
    ↓
Get per-person violence score
    ↓ [Stage 3: Fusion]
Aggregate scores → Frame verdict
    ↓
Annotate & Save Output Video
```

### Evaluation Pipeline
```
Random Sample (10 violence + 10 non-violence videos)
    ↓ [eval_sample.py]
Run full inference on each
    ↓
Collect predictions
    ↓
Compare with ground truth labels
    ↓
Generate metrics & confusion matrix
```

---

## File Dependencies

### violence_detector.py
- **Imports from**: TensorFlow, Keras, OpenCV, Ultralytics YOLO, NumPy
- **Defines**: 
  - `create_cnn_lstm()` - Model architecture
  - `ViolenceDetector` class - Main detection engine
  - Constants: `SEQUENCE_LENGTH=20`, `IMG_H/W=64`, `YOLO_CONF=0.5`, `VIOLENCE_THRESH=0.5`
- **Used by**: `pipeline.py`, `eval_sample.py`, `train.py`, Jupyter notebooks

### pipeline.py
- **Imports from**: `violence_detector.py`, OpenCV, TensorFlow
- **Modes**:
  - `--mode full` - Extract data → Train → Infer
  - `--mode train` - Training only
  - `--mode infer` - Inference only (requires --weights)
- **Used by**: Direct command-line execution or notebooks

### train.py
- **Imports from**: `violence_detector.py`, scikit-learn, OpenCV, TensorFlow
- **Functions**:
  - `extract_sequences_from_video()` - Extract 20-frame sequences
  - `load_data_from_folders()` - Load all videos from dataset dirs
  - `create_data_splits()` - Train/val split
  - Model training with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
- **Outputs**: `model_weights.h5`

### eval_sample.py
- **Imports from**: `violence_detector.py`, OpenCV, NumPy, TensorFlow
- **Purpose**: Validate model on random samples
- **Outputs**: Per-video predictions, metrics, confusion matrix

### Jupyter Notebooks
- **main.ipynb**: 
  - Interactive training workflow
  - Data exploration
  - Visualization of training curves
  - Model evaluation
  
- **attempt_2.ipynb**:
  - Alternative approaches
  - Experimental configurations
  - Ablation studies

---

## Key Constants & Configuration

Located in `violence_detector.py`:
```python
SEQUENCE_LENGTH = 20        # Frames per person sequence buffer
IMG_H, IMG_W   = 64, 64    # Crop dimensions for CNN
YOLO_CONF      = 0.5       # Minimum YOLO detection confidence
VIOLENCE_THRESH = 0.5      # Score threshold for VIOLENT classification
```

Training Parameters (in `train.py`):
```python
--epochs 30
--batch_size 4
--val_split 0.15
--max_seqs_per_video 3
```

---

## Model I/O Specifications

### CNN+LSTM Model
| Property | Value |
|----------|-------|
| Input Shape | (batch_size, 20, 64, 64, 3) |
| Output Shape | (batch_size, 1) |
| Output Range | [0, 1] (sigmoid) |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |

### YOLOv8-nano
| Property | Value |
| -------- | ----- |
| Input | RGB images (any size) |
| Output | Bounding boxes (x, y, w, h) + confidence |
| Classes | Person (class 0) |
| Inference Speed | ~5ms per frame (GPU) |

---

## Execution Flows

### Flow 1: Full Pipeline (Training → Inference)
```python
python pipeline.py --mode full --input test.mp4
```
1. Extract sequences from Violence/ and NonViolence/
2. Train CNN+LSTM model
3. Run inference on input video
4. Save annotated output

### Flow 2: Training Only
```python
python train.py --epochs 30 --batch_size 4
```
1. Load all video folders
2. Extract sequences with augmentation
3. Train with early stopping
4. Save best model weights

### Flow 3: Inference Only
```python
python violence_detector.py --input video.mp4 --weights model_weights.h5
```
1. Load trained model
2. Process video frame-by-frame
3. Track persons and classify violence
4. Output annotated video

### Flow 4: Evaluation
```python
python eval_sample.py
```
1. Random sample 10 violence + 10 non-violence
2. Run full pipeline on each
3. Compute metrics

---

## Extension Points

### Adding New Detection Models
Modify `violence_detector.py`:
- Replace `create_cnn_lstm()` with alternative architecture
- Update model loading/saving logic

### Adding New Data Sources
Modify `train.py`:
- Add new folder paths in `load_data_from_folders()`
- Implement custom data augmentation

### Custom Inference Modes
Modify `pipeline.py`:
- Add new `--mode` options
- Implement custom prediction logic

### Real-time Streaming
Extend `violence_detector.py`:
- Replace file I/O with camera/stream input
- Optimize frame buffering

---

## Performance Considerations

| Component | Optimization |
|-----------|--------------|
| YOLO Detection | Uses lightweight nano variant |
| Sequence Buffering | 20-frame sliding window |
| CNN Processing | TimeDistributed for efficiency |
| GPU Acceleration | TensorFlow auto-uses GPU if available |

---

## Testing & Validation

**Unit Testing Locations**:
- `main.ipynb` - Model architecture validation
- `attempt_2.ipynb` - Alternative approaches testing
- `eval_sample.py` - End-to-end pipeline validation

**Metrics Tracked**:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Per-class performance

---

## Dependencies Graph

```
violence_detector.py
├── TensorFlow/Keras
├── Ultralytics YOLO
├── OpenCV
└── NumPy

pipeline.py
├── violence_detector.py
├── TensorFlow
└── OpenCV

train.py
├── violence_detector.py
├── TensorFlow
├── scikit-learn
└── OpenCV

eval_sample.py
├── violence_detector.py
└── OpenCV
```

---

**Last Updated**: May 2026
