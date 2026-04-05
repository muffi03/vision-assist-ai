from ultralytics import YOLO
import cv2
import pyttsx3
import time
import torch
import numpy as np
import threading
import queue

# -------------------------------
# Simple CV heuristics for navigation
# -------------------------------

def detect_doors(frame):
    """Detect door-like vertical rectangles using edge + Hough lines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=80, maxLineGap=10)

    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if abs(x1-x2) < 10 and abs(y1-y2) > 80:
                vertical_lines.append((x1,y1,x2,y2))

    # door heuristic: two vertical lines roughly parallel
    if len(vertical_lines) >= 2:
        return True

    return False


def detect_stairs(frame):
    """Detect stair-like horizontal patterns."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=60, maxLineGap=5)

    horizontal_count = 0

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if abs(y1-y2) < 10 and abs(x1-x2) > 60:
                horizontal_count += 1

    # stairs usually produce multiple horizontal edges
    if horizontal_count >= 4:
        return True

    return False

speech_queue = queue.Queue()


class VideoStream:
    def __init__(self, src=0):
        # Force macOS built‑in camera using AVFoundation backend
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
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
        # If frame has not been captured yet, return failure safely
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()


# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# -------------------------------
# Vision Assist AI v7
# Detection + Tracking + Depth
# Scene understanding + Navigation
# -------------------------------

USE_PHONE_CAMERA = False
CAMERA_INDEX = 1  # 0 may select Continuity Camera, 1 usually selects MacBook webcam
PHONE_STREAM_URL = "http://192.168.1.2:8080/video"
FRAME_SKIP = 3

important_objects = [
    "person","chair","car","bench","bottle","dog","cat","backpack","handbag"
]

navigation_structures = ["stairs","door"]

priority_obstacles = ["person","car","chair","bench"]
priority_navigation = ["door", "stairs", "step", "entrance"]

# Simple navigation memory
path_memory = []


def initialize_models():

    detector = YOLO("yolov8n.pt")
    detector.to(DEVICE)

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(DEVICE)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    depth_transform = transforms.small_transform

    # segmentation removed for performance (not used in navigation logic)

    return detector, midas, depth_transform


def initialize_speech():

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    return engine


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


def speak(engine, text):
    speech_queue.put(text)


def initialize_camera():

    if USE_PHONE_CAMERA:
        stream = VideoStream(PHONE_STREAM_URL).start()
    else:
        stream = VideoStream(CAMERA_INDEX).start()

    return stream


def estimate_direction(center_x, width):

    if center_x < width * 0.33:
        return "left"
    elif center_x > width * 0.66:
        return "right"
    else:
        return "center"


def main():

    print("Vision Assist AI v7 started — press q to quit")

    detector, midas, depth_transform = initialize_models()
    engine = initialize_speech()

    # start single speech worker thread
    threading.Thread(target=speech_worker, args=(engine,), daemon=True).start()

    stream = initialize_camera()

    last_spoken = {}
    cooldown = 3

    frame_count = 0

    depth_map = None

    while True:
        try:

            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 360))

            frame_count += 1

            if frame_count % FRAME_SKIP != 0:
                cv2.imshow("Vision Assist AI v7", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            height, width, _ = frame.shape

            # Prepare RGB image once per loop
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------- Door / Stair Detection (heuristic CV) --------
            door_detected = detect_doors(frame)
            stairs_detected = detect_stairs(frame)

            # -------- Detection + Tracking --------
            results = detector(frame, imgsz=480, verbose=False)

            if results is not None and len(results) > 0:
                annotated = results[0].plot()
            else:
                annotated = frame

            # -------- Depth Estimation (run less frequently) --------
            if frame_count % 6 == 0:
                input_batch = depth_transform(img).to(DEVICE)

                with torch.inference_mode():
                    depth = midas(input_batch)
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(1),
                        size=img.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze()

                depth_map = depth.cpu().numpy()

            # Scene segmentation disabled for performance

            obstacle_detected = False

            # Prioritize structural navigation cues
            if door_detected:
                speak(engine, "doorway detected ahead")

            if stairs_detected:
                speak(engine, "stairs detected ahead, proceed carefully")

            if results is None or len(results) == 0 or results[0].boxes is None:
                boxes_iter = []
            else:
                boxes_iter = results[0].boxes

            for box in boxes_iter:

                cls = int(box.cls[0])
                label = detector.names[cls]

                if label not in important_objects and label not in navigation_structures:
                    continue

                x1,y1,x2,y2 = box.xyxy[0]

                cx = int((x1+x2)/2)
                cy = int((y1+y2)/2)

                cx = max(0,min(cx,width-1))
                cy = max(0,min(cy,height-1))

                if depth_map is not None and cy < depth_map.shape[0] and cx < depth_map.shape[1]:
                    depth_val = depth_map[cy][cx]
                else:
                    depth_val = 0

                if depth_val > 7:
                    distance="very close"
                elif depth_val > 4:
                    distance="close"
                else:
                    distance="ahead"

                direction = estimate_direction(cx,width)

                # Floor / safe path estimation
                if depth_map is not None and cy > int(height * 0.7):
                    # objects detected near bottom of frame likely block walking path
                    if depth_val > 6:
                        obstacle_detected = True

                current_time = time.time()

                message=None
                guidance=None

                # Navigation structure detection (high priority)
                if label in navigation_structures or label in priority_navigation:
                    message = f"{label} ahead"
                    guidance = "possible passage or step detected"

                # Obstacle detection
                elif label in priority_obstacles and distance=="very close":

                    obstacle_detected=True

                    if direction=="center":
                        message="Obstacle ahead"
                        guidance="move slightly left or right"
                    elif direction=="left":
                        message="Obstacle on your left"
                        guidance="move right"
                    else:
                        message="Obstacle on your right"
                        guidance="move left"

                else:
                    message=f"{label} {distance} {direction}"

                if label not in last_spoken or current_time-last_spoken[label]>cooldown:

                    speak(engine,message)

                    if guidance:
                        speak(engine,guidance)

                    last_spoken[label]=current_time

            # -------- Path Clearance --------
            # only announce path clear occasionally
            if not obstacle_detected and frame_count % 20 == 0:
                speak(engine,"path clear")

            # -------- Navigation Memory --------
            path_memory.append(obstacle_detected)

            if len(path_memory)>50:
                path_memory.pop(0)

            cv2.imshow("Vision Assist AI v7",annotated)

            if cv2.waitKey(1) & 0xFF==ord("q"):
                break
        except Exception as e:
            print("Loop error:", e)
            continue

    speech_queue.put(None)

    stream.stop()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()