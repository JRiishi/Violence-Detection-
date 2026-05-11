# Violence Detection System

A comprehensive end-to-end deep learning pipeline for real-time violence detection in videos using object detection and temporal action recognition.

## Overview

This project implements a three-stage violence detection system that:

1. **Stage 1 (Detection)**: Uses YOLOv8-nano to detect and track persons in video frames
2. **Stage 2 (Classification)**: Applies a CNN+LSTM model to classify violence per tracked person
3. **Stage 3 (Fusion)**: Aggregates per-person predictions to generate frame-level violence verdicts

## Architecture

### Stage 1: Person Detection & Tracking
- **Model**: YOLOv8-nano (lightweight, fast)
- **Purpose**: Detects all persons in each frame and maintains consistent tracking across frames
- **Output**: Bounding boxes with person IDs

### Stage 2: Violence Classification (CNN+LSTM)
- **Architecture**: 
  - TimeDistributed CNN (2 Conv2D layers with MaxPooling, BatchNormalization, Dropout)
  - LSTM layer (128 units) for temporal pattern recognition
  - Dense output layers (64 → 32 → 1 neuron with sigmoid)
- **Input**: 20-frame sequences of 64×64 cropped person regions
- **Output**: Binary violence probability per person sequence

### Stage 3: Fusion & Verdict
- Aggregates per-person violence scores to generate final frame-level classification
- Configurable violence threshold (default: 0.5)

## Project Structure

```
Violence_Detector/
├── violence_detector.py      # Main detection pipeline + model definition
├── pipeline.py               # End-to-end training & inference workflow
├── train.py                  # Training script for CNN+LSTM model
├── eval_sample.py            # Evaluation on sample videos
├── Violence/                 # Training data: violent videos
├── Violence_2/               # Additional violent videos
├── NonViolence/              # Training data: non-violent videos
├── NonViolence_2/            # Additional non-violent videos
├── model_weights.h5          # Trained model weights
├── yolov8n.pt                # YOLOv8-nano pre-trained weights
├── main.ipynb                # Jupyter notebook for interactive exploration
├── attempt_2.ipynb           # Additional notebook for experiments
├── eval_sample.py            # Evaluation script
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11+ (for GPU acceleration, optional but recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/JRiishi/Violence-Detection-.git
cd Violence_Detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- `ultralytics` - YOLOv8 implementation
- `tensorflow` / `keras` - Deep learning framework
- `opencv-python` - Video processing
- `numpy`, `scikit-learn` - Data processing

## Usage

### 1. Training the Model

```bash
# Train CNN+LSTM on extracted sequences
python train.py \
    --data_dir . \
    --epochs 30 \
    --batch_size 4 \
    --output model_weights.h5 \
    --val_split 0.15 \
    --max_seqs_per_video 3
```

### 2. Running Full Pipeline (Extract → Train → Infer)

```bash
python pipeline.py \
    --mode full \
    --input test_video.mp4 \
    --output results.mp4 \
    --weights model_weights.h5
```

### 3. Inference Only (Using Trained Model)

```bash
python violence_detector.py \
    --input test_video.mp4 \
    --output results_annotated.mp4 \
    --weights model_weights.h5
```

### 4. Evaluate on Sample Videos

```bash
python eval_sample.py \
    --violence Violence \
    --nonviolence NonViolence \
    --weights model_weights.h5 \
    --yolo yolov8n.pt \
    --n 10
```

## Key Features

- **Lightweight**: Uses YOLOv8-nano for efficient person detection
- **Temporal Reasoning**: LSTM captures violence patterns across frames
- **Scalable**: Process multiple tracked persons per frame independently
- **Flexible**: Train on custom violence/non-violence datasets
- **Video Support**: Works with multiple formats (MP4, AVI, MOV, MKV, etc.)

## Model Configuration

Key hyperparameters in `violence_detector.py`:

```python
SEQUENCE_LENGTH = 20        # Frames per person sequence
IMG_H, IMG_W   = 64, 64    # Crop dimensions
YOLO_CONF      = 0.5       # YOLO confidence threshold
VIOLENCE_THRESH = 0.5      # Violence classification threshold
```

## Data Format

### Training Data Structure
```
Violence/
    ├── video1.mp4
    ├── video2.avi
    └── ...
NonViolence/
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

### Sequence Extraction
- Extracts non-overlapping 20-frame sequences from each video
- Resizes frames to 64×64 pixels
- Normalizes pixel values to [0, 1]
- Supports multiple sequences per video

## Performance Metrics

The model is evaluated on:
- **Binary classification accuracy** (violence vs. non-violence)
- **Precision & Recall** per class
- **Per-frame prediction accuracy** on sample videos
- **Inference speed** (frames per second)

## Training Results

Details about training can be found in:
- `main.ipynb` - Interactive training logs and visualizations
- `attempt_2.ipynb` - Additional experiments and ablations
- Model checkpoint: `model_weights.h5`

## Notebooks

- **main.ipynb**: Primary notebook for training, visualization, and analysis
- **attempt_2.ipynb**: Secondary notebook for alternative approaches and experiments

## Output Format

For a processed video:
- **Annotated frames** with:
  - Bounding boxes around detected persons
  - Person IDs (tracking)
  - Per-person violence scores
  - Frame-level verdict (VIOLENT / SAFE)
- **Output video** saved in same format as input

## Troubleshooting

### No persons detected
- Lower `YOLO_CONF` threshold in `violence_detector.py`
- Ensure video quality is sufficient
- Check lighting conditions

### Model predictions not making sense
- Ensure `model_weights.h5` is properly trained
- Verify training data quality
- Check sequence length and image dimensions match

### Out of memory errors
- Reduce `batch_size` in training
- Reduce number of frames processed at once
- Use GPU with sufficient VRAM

## Future Improvements

- [ ] Multi-action classification (not just binary violence)
- [ ] Attention mechanisms for interpretability
- [ ] Ensemble methods combining multiple models
- [ ] Real-time streaming support
- [ ] Web API interface
- [ ] Model optimization (quantization, pruning)

## License

[Specify your license here]

## References

- YOLOv8: https://github.com/ultralytics/ultralytics
- CNN-LSTM for Action Recognition: [Relevant Papers]
- TensorFlow/Keras Documentation: https://www.tensorflow.org/

## Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: May 2026
