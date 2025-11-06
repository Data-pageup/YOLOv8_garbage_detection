# YOLOv8 Object Detection: Input Size Comparison Lab Report
## Garbage Detection Dataset

### 1. Dataset Summary

**Dataset Name:** Garbage Detection (Roboflow)

**Dataset Statistics:**
- Total Images: 1,255
- Number of Classes: 1
  - garbage: All types of garbage/trash

**Data Split:**
- Training: 1,155 images (92%)
- Validation: 50 images (4%)
- Testing: 50 images (4%)

**Key Dataset Characteristics:**
- Format: YOLOv8 format (YOLO11 compatible)
- Image Resolution: Variable
- Annotation Format: Bounding boxes with class labels
- Challenge: Single-class dataset; requires focus on detection across varying garbage types and contexts

---

### 2. Experimental Setup

#### 2.1 Model Architecture
- Model: YOLOv8 Nano (YOLOv8n)
- Framework: Ultralytics YOLOv8
- Pre-trained Weights: YOLOv8 COCO pretrained
- Hardware: GPU (NVIDIA Tesla T4, 15GB VRAM)
- Batch Size: 32

#### 2.2 Experimental Configuration

**Baseline Training (416×416):**
- Input Size: 416×416 pixels
- Epochs: 10
- Training Time: 1.99 minutes
- Device: CUDA (GPU)

**Experiment 1 (608×608):**
- Input Size: 608×608 pixels
- Epochs: 10
- Training Time: 3.23 minutes
- Device: CUDA (GPU)

---

### 3. Results

#### 3.1 Performance Metrics Comparison

| Metric | 416×416 | 608×608 | Change |
|--------|---------|---------|--------|
| mAP@0.5 | 0.3520 | 0.3630 | +0.0110 (+3.1%) |
| mAP@0.5:0.95 | 0.1420 | 0.1490 | +0.0070 (+4.9%) |
| Precision | 0.5550 | 0.4900 | -0.0650 (−11.7%) |
| Recall | 0.3220 | 0.4000 | +0.0780 (+24.2%) |
| Inference Time (ms/image) | 1.3 | 2.6 | +1.3 (100% slower) |
| Model Size (MB) | 6.2 | 6.2 | — |
| Training Time (minutes) | 1.99 | 3.23 | +1.24 (+62.3%) |

---

#### 3.2 Detailed Metrics (Per-Class)

**Baseline (416×416):**
- Precision: 0.5563
- Recall: 0.3217
- mAP@0.5: 0.3520
- mAP@0.5:0.95: 0.1427

**Experiment (608×608):**
- Precision: 0.4895
- Recall: 0.4000
- mAP@0.5: 0.3624
- mAP@0.5:0.95: 0.1490

---

### 4. Analysis & Discussion

#### 4.1 Key Findings

**Recall Improvement (+24.25%):**
- 608×608 detects significantly more garbage objects
- Important when missing garbage is problematic
- Lower recall in 416×416 means some garbage is not detected

**Precision Trade-off (−11.7%):**
- 608×608 has more false positives
- Better detection rate but more false alarms

**mAP Improvement (+3.1%):**
- Higher accuracy at both mAP@0.5 and mAP@0.5:0.95

**Inference Speed Impact (+100%):**
- 608×608 doubles inference time
- Still real-time on GPU

**Training Cost Increase (+62.3%):**
- 608×608 requires more training time
- Model size remains identical

---

#### 4.2 Input Size Impact Analysis

**416×416 Advantages:**
- Faster inference
- Higher precision
- Faster training
- Better for resource-constrained devices

**608×608 Advantages:**
- Better recall
- Higher overall accuracy
- More robust to object scale variations
- Better for accuracy-critical applications

---

#### 4.3 GPU vs CPU Comparison

**Training Time:**
- GPU (T4):
  - 416×416: 1.99 minutes
  - 608×608: 3.23 minutes
- Estimated CPU Time:
  - 30–50 minutes (15–25× slower)

**Inference Speed:**
- GPU: 1.3–2.6 ms/image
- CPU: 100–200 ms/image (not real-time)

---

### 5. Qualitative Analysis

**Observations:**
- Both models successfully detect garbage
- 608×608 shows larger bounding boxes
- 608×608 detects more small objects
- 416×416 has fewer false positives
- Differences small visually, but real in metrics

**Sample Detections:**
1. Trash bins with multiple garbage items → both detect well
2. Loose garbage → 608×608 better
3. Garbage in landfill → similar performance
4. Urban litter → 608×608 better
5. Mixed debris → similar, slight advantage for 608×608

---

### 6. Conclusion

#### Summary

This lab compared YOLOv8 performance across two input sizes (416×416 and 608×608). Key findings:

1. Larger input size (608×608) provides better detection capability (+24.2% recall).
2. Input size creates a speed–accuracy trade-off:
   - 416×416: faster, fewer false positives
   - 608×608: slower, catches more garbage
3. GPU acceleration is essential for practical experimentation.

---

### Recommendations

**For Production Deployment:**
Use 608×608 with GPU for maximum accuracy.

**For Edge Devices:**
Use 416×416 for faster inference.

**For Mobile Applications:**
Use model quantization.

**Future Work:**
- Test 512×512
- Experiment with YOLOv8s, YOLOv8m
- Model pruning and quantization
- Benchmark CPU performance
- Collect more diverse garbage images

---

### Impact Summary

- Detection Rate: +24.2% improvement with 608×608
- Accuracy: +3.1% improvement in mAP@0.5
- Speed Cost: +100% inference time
- Training Cost: +62.3% more time
- Deployment: Both sizes viable depending on requirements

