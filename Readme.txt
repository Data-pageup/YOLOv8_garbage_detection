🗑️ YOLOv8 Garbage Detection — Input Size Comparison

This repository contains a full lab report and results for YOLOv8-based garbage detection, comparing two input resolutions (416×416 vs. 608×608) to measure accuracy, recall, inference speed, and training cost.

If you need the YOLOv8 training notebook (.ipynb), contact me here:
📧 amirthaganeshramesh@gmail.com

📦 Dataset Summary

Dataset: Garbage Detection (Roboflow)
Total Images: 1,255
Classes: 1 (garbage)

Split:

Training: 1,155 (92%)

Validation: 50 (4%)

Testing: 50 (4%)

Key Characteristics:

YOLOv8/YOLO11-compatible format

Varying image resolutions

Bounding-box annotations

Single-class dataset → higher importance of recall

⚙️ Experimental Setup
✅ Model Architecture

Model: YOLOv8n (Nano)

Framework: Ultralytics YOLOv8

Pretrained Weights: COCO

Hardware: NVIDIA Tesla T4 (15GB VRAM)

Batch Size: 32

✅ Training Configurations
Baseline — 416×416

Epochs: 10

Training Time: 1.99 min

Device: GPU

Experiment — 608×608

Epochs: 10

Training Time: 3.23 min

Device: GPU

📊 Results
🔍 Performance Metrics Comparison
Metric	416×416	608×608	Change
mAP@0.5	0.3520	0.3630	+3.1%
mAP@0.5:0.95	0.1420	0.1490	+4.9%
Precision	0.5550	0.4900	−11.7%
Recall	0.3220	0.4000	+24.2%
Inference Time	1.3 ms	2.6 ms	+100%
Training Time	1.99 min	3.23 min	+62.3%
✅ Key Observations

608×608 catches more garbage objects (higher recall).

416×416 has fewer false positives (higher precision).

608×608 improves mAP but costs double inference time.

Both are real-time on GPU; CPU is too slow for deployment.

🧠 Analysis & Discussion
⭐ Advantages of 416×416

Fastest inference

More precise (fewer false alarms)

Best for edge devices / low compute

⭐ Advantages of 608×608

Detects more garbage

Higher mAP

More robust to scale variations

Best for accuracy-critical applications

🚀 GPU vs CPU

GPU training: 2–3 minutes

CPU training: 30–50 minutes (15–25× slower)

GPU inference: 380–770 FPS

CPU inference: not real-time

🖼️ Qualitative Results

Observed trends:

608×608 detects more small objects

416×416 produces fewer false positives

Both perform well on mixed garbage scenes

✅ Conclusion
Final Takeaways

608×608 improves detection rate by +24.2%

416×416 is faster but slightly less accurate

GPU acceleration is essential for practical experimentation

Best input size depends on your deployment needs

✅ Recommendations
For Production Deployment

Use 608×608 (higher recall + better accuracy).

For Edge Devices

Use 416×416 (speed-optimized).

For Mobile Apps

Use quantized models (8-bit or FP16).

For Future Work

Try input size 512×512 (middle ground)

Test larger YOLOv8 models (s/m/l/x)

Add pruning + quantization

Expand dataset

📧 Need the YOLOv8 Notebook (.ipynb)?

Contact: amirthaganeshramesh@gmail.com
