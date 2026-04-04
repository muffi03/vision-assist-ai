from ultralytics import YOLO
import cv2
import pyttsx3

engine = pyttsx3.init()

model = YOLO("yolov8n.pt")

cap=cv2.VideoCapture(0)

spoken_objects = set()

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label not in spoken_objects:
            print(label)
            engine.say(label + " detected")
            engine.runAndWait()
            spoken_objects.add(label)

    cv2.imshow("Vision Assist AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()