# 🚘 RoadWatch Copilot

> Real-time Computer Vision Driver Assistance System for road scene understanding, vehicle tracking, lane perception, and collision-risk analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLO11-Object%20Detection-purple)
![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-orange)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

---

## 📌 Overview

**RoadWatch Copilot** is a real-time Advanced Driver Assistance System (ADAS) prototype built with Computer Vision and Deep Learning.

The system analyzes video captured from a front-facing vehicle camera to understand the surrounding road environment and provide contextual driver warnings.

The project focuses on:

- Vehicle and pedestrian detection
- Multi-object tracking
- Lane detection
- Lane departure analysis
- Vehicle distance estimation
- Time-To-Collision (TTC) estimation
- Forward collision warning
- Traffic sign recognition
- Context-aware warning prioritization
- Real-time edge inference

> **Important:** RoadWatch Copilot is a research/portfolio prototype.  
> It only provides driver-assistance warnings and does **not** control steering, braking, or acceleration.

---

# 🎯 Project Motivation

Modern ADAS systems combine multiple perception modules rather than relying on a single neural network.

RoadWatch Copilot is designed to explore this complete perception pipeline:

```text
Front Camera
     │
     ▼
Video Stream
     │
     ├───────────────┐
     │               │
     ▼               ▼
Object Detection   Lane Detection
     │               │
     ▼               ▼
Object Tracking   Lane Geometry
     │               │
     └───────┬───────┘
             ▼
      Scene Understanding
             │
     ┌───────┴─────────┐
     ▼                 ▼
Distance Estimation   Lane Position
     │                 │
     ▼                 ▼
Relative Motion      Lane Departure
     │
     ▼
Time-To-Collision
     │
     ▼
Risk Engine
     │
     ▼
Warning Manager
     │
     ▼
Visual / Voice Alert
```

The goal is not only to run object detection, but to build a modular real-time perception system where multiple Computer Vision components work together.

---

# ✨ Current Features

## ✅ Object Detection

RoadWatch uses **YOLO11** for real-time road object detection.

Currently supported COCO classes include:

| Object     | Detection |
| ---------- | --------- |
| Person     | ✅        |
| Bicycle    | ✅        |
| Car        | ✅        |
| Motorcycle | ✅        |
| Bus        | ✅        |
| Truck      | ✅        |

Example output:

```text
car ID: 4      confidence: 0.91
truck ID: 7    confidence: 0.87
person ID: 12  confidence: 0.82
```

---

## ✅ Multi-Object Tracking

The project integrates **ByteTrack** with YOLO detections.

Each detected road user receives a persistent tracking ID across consecutive video frames.

```text
Frame 120 → Car ID #7
Frame 121 → Car ID #7
Frame 122 → Car ID #7
Frame 123 → Car ID #7
```

Tracking is important for later modules such as:

- motion analysis
- trajectory estimation
- relative speed estimation
- distance change
- Time-To-Collision

---

## ✅ Track History

RoadWatch stores recent center positions for tracked objects:

```text
Track ID #7

Frame t-3 → (430, 320)
Frame t-2 → (437, 326)
Frame t-1 → (445, 333)
Frame t   → (454, 341)
```

This information can later be used for motion and risk estimation.

---

## ✅ Classical Lane Detection Baseline

The current lane detection prototype uses a traditional Computer Vision pipeline:

```text
Input Frame
    │
    ▼
HLS Color Filtering
    │
    ├── White Lane Mask
    └── Yellow Lane Mask
    │
    ▼
Gaussian Blur
    │
    ▼
Canny Edge Detection
    │
    ▼
Region of Interest
    │
    ▼
Hough Line Transform
    │
    ▼
Slope Filtering
    │
    ├── Left Lane Candidates
    └── Right Lane Candidates
    │
    ▼
Lane Estimation
```

This implementation serves as a **classical CV baseline**.

### Current limitation

Hough-based lane detection can become unstable under:

- faded lane markings
- curved roads
- shadows
- occlusion
- poor lighting
- road surfaces with weak lane boundaries

For this reason, a Deep Learning based lane detector is planned as the next stage.

---

## 🚧 Deep Learning Lane Detection

**Status: In Progress**

The classical lane detector will be replaced or complemented by a neural lane detection model.

The goal is to improve robustness under real-world driving conditions while maintaining real-time inference performance.

---

# 🧠 System Architecture

```text
                        ┌─────────────────┐
                        │  Front Camera   │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   VideoReader   │
                        └────────┬────────┘
                                 │
                                 ▼
                              Frame
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
        ┌─────────────────┐           ┌─────────────────┐
        │     YOLO11      │           │  Lane Detector  │
        │ Object Detector │           │                 │
        └────────┬────────┘           └────────┬────────┘
                 │                             │
                 ▼                             ▼
        ┌─────────────────┐             Lane Geometry
        │    ByteTrack    │                    │
        └────────┬────────┘                    │
                 │                             │
                 ▼                             │
          Tracked Objects                      │
                 │                             │
                 └─────────────┬───────────────┘
                               ▼
                     ┌───────────────────┐
                     │ Scene Perception  │
                     └─────────┬─────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
            Distance          Lane             TTC
           Estimation       Departure       Estimation
               │               │               │
               └───────────────┼───────────────┘
                               ▼
                       ┌───────────────┐
                       │  Risk Engine  │
                       └───────┬───────┘
                               ▼
                      ┌─────────────────┐
                      │ Warning Manager │
                      └────────┬────────┘
                               ▼
                    Visual / Voice Warning
```

---

# 🗂️ Project Structure

```text
roadwatch-copilot/
│
├── configs/
│   ├── app.yaml
│   ├── camera.yaml
│   ├── detection.yaml
│   └── warning.yaml
│
├── data/
│   ├── raw/
│   │   ├── videos/
│   │   └── images/
│   ├── processed/
│   └── output/
│       ├── videos/
│       ├── images/
│       └── logs/
│
├── models/
│   ├── detection/
│   ├── lane/
│   ├── traffic_sign/
│   └── depth/
│
├── src/
│   └── roadwatch/
│       │
│       ├── core/
│       │   ├── pipeline.py
│       │   ├── types.py
│       │   └── video_reader.py
│       │
│       ├── detection/
│       │   ├── vehicle_detector.py
│       │   ├── pedestrian_detector.py
│       │   └── traffic_sign_detector.py
│       │
│       ├── tracking/
│       │   └── vehicle_tracker.py
│       │
│       ├── lane/
│       │   ├── lane_detector.py
│       │   └── lane_departure.py
│       │
│       ├── distance/
│       │   ├── distance_estimator.py
│       │   └── calibration.py
│       │
│       ├── risk/
│       │   ├── ttc.py
│       │   ├── collision.py
│       │   └── risk_engine.py
│       │
│       ├── signs/
│       │   ├── recognizer.py
│       │   └── speed_limit.py
│       │
│       ├── warning/
│       │   ├── warning_manager.py
│       │   ├── priority.py
│       │   └── audio.py
│       │
│       ├── visualization/
│       │   └── hud.py
│       │
│       └── utils/
│           ├── fps.py
│           ├── geometry.py
│           └── logger.py
│
├── backend/
├── frontend/
├── deployment/
│   ├── onnx/
│   ├── tensorrt/
│   └── docker/
│
├── tests/
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

The project follows a modular architecture so perception, tracking, risk estimation, warning logic, visualization, and deployment can evolve independently.

---

# 🛠️ Tech Stack

### Computer Vision & AI

- Python
- OpenCV
- NumPy
- Ultralytics YOLO11
- ByteTrack
- PyTorch

### Planned AI Components

- Deep Learning Lane Detection
- Monocular Depth Estimation
- Traffic Sign Recognition
- TTC / collision-risk estimation

### Backend & Visualization

Planned:

- FastAPI
- React
- Real-time driving HUD

### Edge Deployment

Planned:

- ONNX
- TensorRT
- Docker
- GPU / Edge inference optimization

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone https://github.com/<your-username>/roadwatch-copilot.git

cd roadwatch-copilot
```

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

# ▶️ Running the Project

Place a test driving video in:

```text
data/raw/videos/test.mp4
```

Run:

```bash
PYTHONPATH=src python -m roadwatch.main
```

Press:

```text
q
```

to exit.

---

# 📊 Current Performance

Initial development test:

| Metric                |            Current Result |
| --------------------- | ------------------------: |
| Input resolution      |                1280 × 720 |
| Input video           |                    30 FPS |
| Observed pipeline FPS |                ~20–27 FPS |
| Object detection      |                   YOLO11n |
| Tracking              |                 ByteTrack |
| Lane baseline         |             Canny + Hough |
| Execution             | Local real-time prototype |

> Performance numbers are preliminary and depend on hardware, input video, and enabled perception modules.

Future benchmarks will include:

- detector latency
- tracker latency
- lane detector latency
- end-to-end latency
- FPS
- GPU/CPU utilization
- detection accuracy
- lane accuracy

---

# 🗺️ Development Roadmap

### Phase 1 — Road Object Perception

- [x] Video processing pipeline
- [x] YOLO11 object detection
- [x] Vehicle detection
- [x] Pedestrian detection
- [x] Real-time HUD

### Phase 2 — Multi-Object Tracking

- [x] ByteTrack integration
- [x] Persistent Track IDs
- [x] Track history
- [x] Trajectory representation

### Phase 3 — Lane Perception

- [x] Classical Canny/Hough baseline
- [x] Region of Interest
- [x] Left/right lane candidate filtering
- [ ] Deep Learning lane detection
- [ ] Temporal lane smoothing
- [ ] Curved lane estimation

### Phase 4 — ADAS Risk Analysis

- [ ] Lane Departure Warning
- [ ] Camera calibration
- [ ] Vehicle distance estimation
- [ ] Relative motion estimation
- [ ] Time-To-Collision
- [ ] Forward Collision Warning

### Phase 5 — Traffic Sign Understanding

- [ ] Traffic sign detection
- [ ] Speed limit recognition
- [ ] Current speed-limit state
- [ ] Speed warning

### Phase 6 — Driver Warning System

- [ ] Risk prioritization
- [ ] Warning cooldown
- [ ] Context-aware alert aggregation
- [ ] Offline voice warning

### Phase 7 — Deployment

- [ ] FastAPI service
- [ ] React dashboard
- [ ] ONNX export
- [ ] TensorRT optimization
- [ ] Docker
- [ ] Edge deployment

---

# ⚠️ Engineering Challenges

RoadWatch Copilot explores several practical Computer Vision challenges.

### Lane Detection

Traditional Hough-based methods are sensitive to road markings, lighting, shadows, road curvature, and camera perspective.

The project therefore uses classical lane detection as a baseline before evaluating Deep Learning approaches.

### Distance Estimation

Estimating physical distance from a single monocular camera is inherently uncertain.

Future implementations will evaluate camera calibration, geometric estimation, and monocular depth models rather than treating raw image-space distance as ground truth.

### Time-To-Collision

TTC requires more than object detection.

A reliable pipeline requires:

```text
Detection
    ↓
Tracking
    ↓
Distance
    ↓
Distance Change
    ↓
Relative Motion
    ↓
TTC
```

This is one reason multi-object tracking is a core component of the project.

---

# 🔬 Planned Experiments

Several approaches will be compared during development:

### Lane Detection

```text
Classical CV
Canny + Hough
      VS
Deep Learning Lane Detection
```

Comparison metrics:

- lane stability
- inference latency
- FPS
- robustness to shadows
- robustness to faded markings
- curved-road performance

### Model Deployment

```text
PyTorch
   ↓
ONNX
   ↓
TensorRT
```

The goal is to evaluate the trade-off between inference speed and model accuracy for real-time edge deployment.

---

# 🔒 Safety Scope

RoadWatch Copilot is designed strictly as a **driver-assistance research prototype**.

The system:

```text
CAN:
✓ Detect road objects
✓ Analyze the driving scene
✓ Estimate potential risks
✓ Display warnings
✓ Generate audio alerts

CANNOT:
✗ Control steering
✗ Apply brakes
✗ Control acceleration
✗ Make autonomous driving decisions
```

Distance and collision-risk outputs should not be treated as production-grade automotive safety measurements without proper sensor calibration, validation, redundancy, and automotive safety engineering.

---

# 🎓 What This Project Demonstrates

RoadWatch Copilot is designed to demonstrate practical skills relevant to **Computer Vision / AI Engineer** roles:

- Real-time Computer Vision
- Object Detection
- Multi-Object Tracking
- Classical Computer Vision
- Deep Learning inference
- Video processing
- Computer Vision pipeline design
- ADAS concepts
- Model integration
- Performance optimization
- Modular Python architecture
- Edge AI deployment
- AI system engineering

---

# 📈 Project Status

```text
Object Detection             ██████████ 100%
Multi-Object Tracking        ██████████ 100%
Classical Lane Detection     ██████████ 100%
Deep Lane Detection          ░░░░░░░░░░   0%
Lane Departure Warning       ░░░░░░░░░░   0%
Distance Estimation          ░░░░░░░░░░   0%
TTC                          ░░░░░░░░░░   0%
Collision Warning            ░░░░░░░░░░   0%
Traffic Sign Recognition     ░░░░░░░░░░   0%
Edge Optimization            ░░░░░░░░░░   0%
```

---

# 👨‍💻 Author

Developed as a Computer Vision / AI Engineering portfolio project.

The project is actively being developed with a focus on real-time perception, ADAS algorithms, and edge AI deployment.

---

# 📄 License

This project is intended for educational, research, and portfolio purposes.
