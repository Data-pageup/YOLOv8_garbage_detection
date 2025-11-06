# YOLOv8 Garbage Detection: Input Size Comparison Lab

A comprehensive experimental study comparing YOLOv8 Nano model performance across different input resolutions for garbage detection using the Roboflow dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Analysis](#analysis)
- [Recommendations](#recommendations)
- [Hardware Performance](#hardware-performance)
- [Future Work](#future-work)
- [Installation](#installation)
- [Usage](#usage)

<img width="303" height="172" alt="image" src="https://github.com/user-attachments/assets/0d267292-7d16-44ac-9b0c-81509ebb9ee0" />


## Overview

This project evaluates how input image resolution impacts object detection performance using YOLOv8 Nano. The study compares two configurations:
- **Baseline:** 416×416 pixels
- **Experiment:** 608×608 pixels

The goal is to understand the speed-accuracy trade-offs and provide guidance for optimal input size selection based on deployment scenarios.

## Dataset

**Garbage Detection Dataset (Roboflow)**

| Property | Value |
|----------|-------|
| Total Images | 1,255 |
| Number of Classes | 1 |
| Class Name | garbage |
| Format | YOLOv8 (YOLO11 compatible) |
| Training Set | 1,155 images (92%) |
| Validation Set | 50 images (4%) |
| Test Set | 50 images (4%) |
| Annotation | Bounding boxes with class labels |

## Experimental Setup

### Model Configuration

```
Model:              YOLOv8 Nano (YOLOv8n)
Framework:          Ultralytics YOLOv8
Pre-trained:        COCO weights
Hardware:           NVIDIA Tesla T4 GPU (15GB VRAM)
Batch Size:         32
Epochs:             10
Optimizer:          SGD
```

### Input Size Configurations

#### Baseline Configuration (416×416)
- **Input Size:** 416×416 pixels
- **Training Time:** 1.99 minutes
- **Device:** CUDA (GPU)

#### Experiment Configuration (608×608)
- **Input Size:** 608×608 pixels
- **Training Time:** 3.23 minutes
- **Device:** CUDA (GPU)

## Results

### Performance Metrics Comparison

| Metric | 416×416 | 608×608 | Change | % Change |
|--------|---------|---------|--------|----------|
| **mAP@0.5** | 0.3520 | 0.3630 | +0.0110 | +3.1% |
| **mAP@0.5:0.95** | 0.1420 | 0.1490 | +0.0070 | +4.9% |
| **Precision** | 0.5550 | 0.4900 | -0.0650 | -11.7% |
| **Recall** | 0.3220 | 0.4000 | +0.0780 | +24.2% |
| **Inference Time (ms)** | 1.3 | 2.6 | +1.3 | +100% |
| **Training Time (min)** | 1.99 | 3.23 | +1.24 | +62.3% |
| **Model Size (MB)** | 6.2 | 6.2 | — | — |

### Per-Class Detailed Metrics

**416×416 Configuration:**
- Precision: 0.5563
- Recall: 0.3217
- mAP@0.5: 0.3520
- mAP@0.5:0.95: 0.1427

**608×608 Configuration:**
- Precision: 0.4895
- Recall: 0.4000
- mAP@0.5: 0.3624
- mAP@0.5:0.95: 0.1490

## Analysis

### Key Findings

#### 608×608 Advantages ✅
- **+24.2% Recall Improvement** - Significantly better at detecting garbage objects
- **+3.1% mAP@0.5 Improvement** - Higher overall accuracy
- **Better Small Object Detection** - Captures loose garbage and small debris
- **Robust to Scale Variations** - Handles objects of different sizes better
- **Suitable for Accuracy-Critical Applications**

#### 416×416 Advantages ✅
- **Faster Inference** - 1.3ms vs 2.6ms per image (50% faster)
- **Higher Precision** - Fewer false positives (55.5% vs 49%)
- **Faster Training** - 1.99 min vs 3.23 min (38% faster)
- **Resource Efficient** - Better for edge devices and mobile deployment
- **Lower Computational Cost**

### Trade-off Analysis

| Factor | 416×416 | 608×608 | Trade-off |
|--------|---------|---------|-----------|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Faster but less accurate |
| Accuracy | ⭐⭐⭐ | ⭐⭐⭐⭐ | More accurate but slower |
| Precision | ⭐⭐⭐⭐ | ⭐⭐⭐ | Higher precision, more false negatives |
| Recall | ⭐⭐⭐ | ⭐⭐⭐⭐ | Better detection rate |
| Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | More resource efficient |

### Qualitative Observations

#### Detection Performance by Scenario

| Scenario | 416×416 | 608×608 | Notes |
|----------|---------|---------|-------|
| Trash bins with multiple items | Good | Better | Larger input helps detect clustering |
| Loose garbage | Misses some | Detects | Better for scattered items |
| Garbage in landfill | Similar | Similar | Both perform adequately |
| Urban litter | Limited | Better | Higher resolution captures small items |
| Mixed debris | Similar | Better | Slight advantage for 608×608 |

## Recommendations

### For Production Deployment 🏭
**Use 608×608 with GPU**
```
- Maximum accuracy (mAP@0.5: 0.3630)
- Best recall rate (40%)
- Real-time on GPU (2.6ms acceptable)
- Ideal when missing detections is costly
```

### For Edge Devices / IoT 🔌
**Use 416×416**
```
- Lower computational requirements
- Faster inference (1.3ms per image)
- Better precision (fewer false alarms)
- Ideal when computational resources are limited
```

### For Mobile Applications 📱
**Use 416×416 with Quantization**
```
- Deploy YOLOv8n-int8 or YOLOv8n-fp16
- Further reduce model size and inference time
- Maintain reasonable accuracy for mobile use
```

### For Custom Requirements 🎯
- **Minimize False Positives:** Use 416×416
- **Minimize False Negatives:** Use 608×608
- **Balance Speed & Accuracy:** Test 512×512 (future work)

## Hardware Performance

### GPU Performance (NVIDIA Tesla T4)

**Training Speed:**
```
416×416:  1.99 minutes for 10 epochs
608×608:  3.23 minutes for 10 epochs
Speedup:  ~1.6× faster with 416×416
```

**Inference Speed:**
```
416×416:  1.3 ms/image   (~769 FPS)
608×608:  2.6 ms/image   (~385 FPS)
Both:     Real-time capable
```

### Estimated CPU Performance

**Training Time (Estimated):**
```
416×416:  30-50 minutes   (15-25× slower than GPU)
608×608:  50-80 minutes   (15-25× slower than GPU)
```

**Inference Speed (Estimated):**
```
416×416:  100-200 ms/image   (not real-time)
608×608:  150-300 ms/image   (not real-time)
```

### Recommendation
**GPU acceleration is essential for practical experimentation and deployment.**

## Future Work

- [ ] Test intermediate resolution (512×512)
- [ ] Experiment with larger models (YOLOv8s, YOLOv8m)
- [ ] Model quantization (INT8, FP16)
- [ ] Comprehensive CPU benchmark
- [ ] Collect more diverse garbage images
- [ ] Domain-specific fine-tuning
- [ ] Real-world deployment testing
- [ ] Deploy on edge devices (NVIDIA Jetson, etc.)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/yolov8-garbage-detection.git
cd yolov8-garbage-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt

```
ultralytics>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
numpy
opencv-python
matplotlib
```

## Usage

### Training

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Train with 416×416 input size
results_416 = model.train(
    data='data.yaml',
    imgsz=416,
    epochs=10,
    batch=32,
    device=0  # GPU device ID
)

# Train with 608×608 input size
results_608 = model.train(
    data='data.yaml',
    imgsz=608,
    epochs=10,
    batch=32,
    device=0
)
```

### Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on image
results = model.predict(
    source='image.jpg',
    imgsz=416,  # or 608
    conf=0.5
)

# Display results
results[0].show()
```

### Batch Inference

```python
# Predict on directory
results = model.predict(
    source='path/to/images/',
    imgsz=416,
    save=True,
    save_txt=True
)
```

## Model Details

| Property | Value |
|----------|-------|
| Model Architecture | YOLOv8 Nano (YOLOv8n) |
| Model Size | 6.2 MB |
| Parameters | ~3.2M |
| Framework | PyTorch |
| Pre-trained Dataset | COCO |
| Output Format | Bounding boxes + confidence |
| Supported Output | Detection, Segmentation, Classification |

## Conclusion

This study demonstrates a clear **speed-accuracy trade-off** in object detection:

1. **Larger Input (608×608)** provides 24.2% better recall—essential for applications where missing garbage detection is costly.

2. **Smaller Input (416×416)** provides 50% faster inference—ideal for real-time constraints on resource-limited devices.

3. **GPU Acceleration** is critical for practical deployment and experimentation.

4. **Both configurations are viable** depending on specific requirements and deployment scenarios.

### Selection Guide

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Maximum Accuracy | 608×608 | Best recall and mAP |
| Real-time Edge Device | 416×416 | Fastest inference |
| Mobile App | 416×416 | Lower resource consumption |
| Server Deployment | 608×608 | GPU-backed, accuracy-first |
| Cost-Sensitive | 416×416 | Cheaper hardware needed |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Garbage Detection Dataset](https://roboflow.com/)
- [YOLOv8 Research Paper](https://arxiv.org/abs/2305.09972)

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Status:** ✅ Complete Lab Report  
**Last Updated:** 2025  
**Author:** Your Name/Team
