import cv2
import numpy as np
import time
from ultralytics import YOLO
import speech

# Load YOLOv11 model
model = YOLO('models/yolov11s.pt')

# Object classes (ensure these match your YOLOv11 model)
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Constants for distance estimation
KNOWN_DISTANCE = 100  # cm
KNOWN_HEIGHT = 50  # cm
FOCAL_LENGTH = (KNOWN_DISTANCE * 200) / KNOWN_HEIGHT

def estimate_depth(top_left, bottom_right):
    pixel_height = bottom_right[1] - top_left[1]
    if pixel_height > 0:
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height
        return distance / 100  # Convert cm to meters
    return -1

def get_object_location(frame_width, x_center):
    if x_center < frame_width * 0.33:
        return "left"
    elif x_center > frame_width * 0.66:
        return "right"
    else:
        return "center"

def search_object(target_object):
    cap = cv2.VideoCapture(0)
    found = False  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_width = frame.shape[1]  
        results = model(frame, show=False, conf=0.4, verbose=False, save=False)

        detected_objects = []  

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = classNames[cls]

                if label == target_object:
                    found = True
                    x_center = (x1 + x2) // 2
                    object_location = get_object_location(frame_width, x_center)
                    distance = estimate_depth((x1, y1), (x2, y2))

                    if distance > 0:
                        detected_objects.append(f"{label} at {object_location}, {distance:.2f} meters away")
                    else:
                        detected_objects.append(f"{label} at {object_location}, distance unknown")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{label} ({distance:.2f}m)", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if detected_objects:
            print(". ".join(detected_objects))
            speech.speak(". ".join(detected_objects))

        cv2.imshow('Object Search', frame)

        if found:
            time.sleep(2)  
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            speech.speak("Exiting object search mode.")
            break

    cap.release()
    cv2.destroyAllWindows()
    

    