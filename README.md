
# Vision Assist AI

Vision Assist AI is a real-time computer vision navigation assistant designed to help visually impaired individuals understand their surroundings using artificial intelligence.

The system combines multiple deep learning models to detect objects, estimate their distance, understand the scene, and provide spoken navigation guidance.

---

## Features

### Real-Time Object Detection
Uses **YOLOv8** to detect objects such as:

- person
- chair
- car
- bench
- backpack
- bottle
- dog

### Object Tracking
Tracks objects across frames to detect moving obstacles.

### Depth Estimation
Uses **MiDaS** to estimate distance from the camera.

Example outputs:

```
person ahead
chair close left
car very close
```

### Obstacle Detection
The system warns the user when an obstacle is close:

```
Obstacle ahead
Obstacle on your left
Obstacle on your right
```

### Navigation Guidance
Guidance instructions are spoken to help navigate around obstacles:

```
move left
move right
move slightly left or right
```

### Voice Feedback
Navigation instructions are spoken using **pyttsx3 text‑to‑speech**.

### Path Clearance Detection
If no obstacle is detected:

```
path clear
```

### Performance Optimizations
The system includes several real‑time optimizations:

- threaded camera capture
- frame skipping
- reduced segmentation frequency
- GPU support (if available)
- asynchronous speech pipeline

---

## AI Models Used

### YOLOv8
Real‑time object detection and tracking.

Model used:

```
yolov8n.pt
```

### MiDaS
Monocular depth estimation from a single RGB image.

Model used:

```
MiDaS_small
```

### DeepLabV3
Semantic scene segmentation.

Model used:

```
deeplabv3_resnet50
```

---

## System Architecture

```
Camera
   ↓
YOLOv8 Detection + Tracking
   ↓
MiDaS Depth Estimation
   ↓
Scene Segmentation
   ↓
Navigation Logic
   ↓
Voice Guidance
```

---

## Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- MiDaS
- DeepLabV3
- pyttsx3

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/vision-assist-ai.git
cd vision-assist-ai
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
python src/main.py
```

Press **Q** to exit.

---

## Future Improvements

Possible future enhancements include:

- spatial audio navigation
- crosswalk and sidewalk detection
- improved scene understanding
- mobile phone deployment
- edge deployment on NVIDIA Jetson

The long‑term goal is to evolve Vision Assist AI into a full assistive navigation system for visually impaired users.
