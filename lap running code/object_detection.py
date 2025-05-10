import cv2
import numpy as np
import time
from ultralytics import YOLO
import speech

# Load YOLO model (Updated for YOLOv11)
model = YOLO('models/yolov11s.pt')

# Define object classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
    "dining table", "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def run_object_detection():
    cap = cv2.VideoCapture(0)
    prev_objects = {}  # Store previously detected objects
    last_speak_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False, conf=0.4, verbose=False, save=False)

        object_counts = {}  # Store detected object counts

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = classNames[cls] if cls < len(classNames) else "unknown"

                # Count detected objects
                object_counts[label] = object_counts.get(label, 0) + 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Check if objects changed and cooldown is over
        if object_counts != prev_objects and time.time() - last_speak_time >= 2:
            # Sort objects by count (highest first)
            sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create speech text in correct format
            speech_text = ", ".join(f"{count} {label}{'s' if count > 1 else ''}" for label, count in sorted_objects) + " detected"

            print(speech_text)
            speech.speak(speech_text)  # Speak detected objects

            prev_objects = object_counts.copy()  # Update previous objects
            last_speak_time = time.time()  # Reset cooldown timer

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speech.speak("Exiting object detection mode.")
            break

    cap.release()
    cv2.destroyAllWindows()
