import cv2
import torch
import numpy as np
import threading
import queue

speech_queue = queue.Queue()


class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

def initialize_models():
    # model initialization code here
    pass

def initialize_camera():
    if USE_PHONE_CAMERA:
        stream = VideoStream(PHONE_STREAM_URL).start()
    else:
        stream = VideoStream(0).start()

    return stream

def speak(engine, text):
    speech_queue.put(text)

def speech_worker(engine):
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            print(text)
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass

def main():
    stream = initialize_camera()
    # other initializations

    engine = initialize_speech()

    # start single speech worker thread
    threading.Thread(target=speech_worker, args=(engine,), daemon=True).start()

    frame_count = 0
    depth_map = None

    while True:
        ret, frame = stream.read()
        if not ret:
            break

        frame_count += 1

        # -------- Depth Estimation (run less frequently) --------
        if frame_count % 4 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = depth_transform(img).to(DEVICE)

            with torch.no_grad():
                depth = midas(input_batch)
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

            depth_map = depth.cpu().numpy()

        # rest of main loop processing

    speech_queue.put(None)
    stream.stop()
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