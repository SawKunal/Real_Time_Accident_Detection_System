# Accident Detection System

A deep learning-based system for detecting traffic accidents in video footage using temporal transformers and EfficientNet features.

## 🎯 Overview

This project implements a real-time accident detection system that analyzes video streams to identify potential traffic accidents. It uses a two-stage approach:
1. **Feature Extraction**: EfficientNet-B1 backbone extracts spatial features from video frames
2. **Temporal Analysis**: Transformer encoder processes temporal sequences to detect accident patterns

## 📊 Performance

- **Window-level Metrics**:
  - AUC: 0.977
  - Average Precision: 0.948
  - F1 Score: 0.867
  - Precision: 0.809
  - Recall: 0.934

- **Event-level Metrics**:
  - Precision: 0.986
  - Recall: 0.688
  - F1 Score: 0.811

## 🏗️ Architecture

### Model Components

1. **Backbone Network**: EfficientNet-B1 (pre-trained on ImageNet)
   - Input: 224×224 RGB frames
   - Output: 1280-dimensional feature vectors
   - Frozen during training for efficiency

2. **Temporal Transformer**:
   - 3-layer transformer encoder
   - 8 attention heads
   - 512 model dimension
   - 2048 FFN dimension
   - Processes sequences of 48 frames (~3 seconds at 16 FPS)

### Data Processing Pipeline

```
Video → Frame Sampling (16 FPS) → Resize (224×224) → EfficientNet → 
Features (1280-D) → Sliding Window (48 frames) → Transformer → 
Accident Probability
```

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
tensorflow>=2.19.0
opencv-python-headless
numpy
pandas
scikit-learn
matplotlib
tqdm
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/accident-detection.git
cd accident-detection

# Install dependencies
pip install -r requirements.txt
```

### Project Structure

```
accident-detection/
├── data/
│   ├── raw/              # Raw video files and labels
│   └── features/         # Extracted features (auto-generated)
├── checkpoints/          # Model checkpoints
├── exports/
│   └── models/          # Final trained models
├── logs/                # Training logs and detection logs
├── eval/                # Evaluation results and metrics
├── Accident.ipynb       # Training notebook
├── test1.py            # Real-time inference script
└── README.md
```

## 📝 Usage

### 1. Prepare Your Data

Organize your videos and labels:

```
data/raw/
├── video1.mp4
├── video1.txt          # Accident timestamps (HH:MM:SS - HH:MM:SS)
├── video2.mp4
└── video2.txt
```

**Label Format** (`video1.txt`):
```
00:01:30 - 00:01:45
00:05:20 - 00:05:35
```

### 2. Training

Open and run the Jupyter notebook:

```bash
jupyter notebook Accident.ipynb
```

The training pipeline includes:
- **Cell 0-1**: Setup and environment configuration
- **Cell 2-5**: Feature extraction from videos
- **Cell 6-7**: Dataset indexing and augmentation
- **Cell 8-11**: Model building and training
- **Cell 12-13**: Evaluation and export

Key hyperparameters:
```python
SEQ_LEN = 48          # Sequence length (frames)
STRIDE = 8            # Sliding window stride
BATCH_SIZE = 32       # Training batch size
TARGET_FPS = 16       # Feature extraction FPS
THRESHOLD = 0.5       # Detection threshold
```

### 3. Real-time Inference

Run the detection script on new videos:

```bash
python test1.py
```

Configure inference parameters in `test1.py`:
```python
VIDEO_PATH = "data/raw/your_video.mp4"
MODEL_WEIGHTS = "transformer_temporal_head.keras"
SEQ_LEN = 48
TARGET_FPS = 16
THRESH = 0.5
PRED_EVERY = 3        # Run inference every N frames
DISPLAY_STRIDE = 2    # Display every N frames
```

**Controls**:
- Press `q` to quit
- Green box: Normal traffic
- Red box: Accident detected

## 🔧 Advanced Features

### Custom Augmentation

The system includes CCTV-optimized augmentations:
- Brightness adjustment (0.8-1.2x)
- Contrast adjustment (0.8-1.25x)
- Gaussian noise (σ=3-12)
- Motion blur (3×3 or 5×5 kernels)
- JPEG compression (quality 40-90)

### Persistent Storage

The system supports persistent storage for tracking detections across sessions:

```python
# Store detection
await window.storage.set('detections:123', JSON.stringify(data))

# Retrieve detection
result = await window.storage.get('detections:123')
```

### Event-Level Detection

The system merges contiguous windows into accident events with configurable parameters:

```python
EVENT_TOLERANCE = 3.0      # Seconds: matching tolerance
MIN_EVENT_DURATION = 0.5   # Seconds: minimum event length
```

## 📈 Evaluation

Run comprehensive evaluation:

```python
# Window-level metrics: precision, recall, F1, AUC
# Event-level metrics: TP, FP, FN for accident events
# Outputs: CSV predictions, JSON results, summary text
```

Results are saved in `eval/`:
- `window_level_predictions.csv`: Per-window predictions
- `window_metrics.json`: Window-level metrics
- `event_results.json`: Event-level results
- `eval_summary.txt`: Human-readable summary

## 🎓 Training Tips

1. **Data Quality**: Ensure accurate timestamp labels
2. **Class Balance**: Use built-in balancing in generator
3. **Learning Rate**: Start with 1e-4, reduce on plateau
4. **Sequence Length**: 48 frames ≈ 3 seconds works well
5. **Stride**: Smaller stride = more samples but slower
6. **Augmentation**: Enable only for limited datasets

## 🔍 Troubleshooting

### Common Issues

**Issue**: Out of memory during training
```python
# Solution: Reduce batch size
BATCH_SIZE = 16  # or 8
```

**Issue**: Features not found
```bash
# Solution: Run feature extraction (Cell 6)
# Ensure videos are in data/raw/
```

**Issue**: Model predicts all zeros
```python
# Solution: Check class balance and threshold
# Try lowering threshold or increasing positive samples
```

**Issue**: Slow inference
```python
# Solution: Adjust prediction frequency
PRED_EVERY = 5  # Predict every 5th frame instead of 3
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{accident_detection_2024,
  author = {Kunal Kumar Saw},
  title = {Accident Detection System},
  year = {2025},
  url = {https://github.com/yourusername/accident-detection}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- EfficientNet paper: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Transformer architecture: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
- Training conducted on Google Colab with T4 GPU

## 📞 Contact
- **Email**: sawkunal556@gmail.com

---

**Note**: This system is designed for research and development purposes. For production deployment, additional testing and validation are recommended.
