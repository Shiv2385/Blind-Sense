import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import time
import speech
import urllib.request  # NEW: Import to fetch frames from ESP32

# Initialize YOLO model
model = YOLO('models/yolov11s.pt')

# ESP32 Camera Stream URL (Replace with your ESP32 IP if needed)
esp32_url = "http://192.168.43.142/cam-mid.jpg"

# Last announcement timestamp
last_label_time = time.time()

# Constants for distance estimation (calibrate for real-world accuracy)
KNOWN_DISTANCE = 100  # cm (measured distance of reference object)
KNOWN_HEIGHT = 50     # cm (actual height of reference object)
FOCAL_LENGTH = (KNOWN_DISTANCE * 200) / KNOWN_HEIGHT  # Example focal length calculation,400

def estimate_distance(top_left, bottom_right):
    """Estimate object distance using known size method."""
    pixel_height = bottom_right[1] - top_left[1]  # Object height in pixels
    if pixel_height > 0:
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height  # Apply distance formula
        return round(distance / 100, 2)  # Convert to meters
    return -1  # Invalid measurement

def save_direction(direction):
    """Save movement direction to a file."""
    with open("dir.txt", "w") as file:
        file.write(direction)

def run_navigation():
    global last_label_time 
    # Since we're using the ESP32, no need to initialize cv2.VideoCapture
    # Instead, we'll fetch the frame via the URL in the loop

    while True:
        try:
            # NEW: Fetch image from ESP32 Camera using URL
            img_resp = urllib.request.urlopen(esp32_url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)  # Decode the image

            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Failed to grab frame from ESP32, retrying...")
                continue
        except Exception as e:
            print(f"Error fetching frame: {e}")
            time.sleep(1)
            continue

        height, width, _ = frame.shape
        roi_width = width // 3  

        # Define ROIs
        left_roi = (0, 0, roi_width, height)
        center_roi = (roi_width, 0, roi_width, height)
        right_roi = (2 * roi_width, 0, roi_width, height)

        # Draw ROIs
        cv2.rectangle(frame, left_roi[:2], (left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]), (0, 0, 255), 2)
        cv2.rectangle(frame, center_roi[:2], (center_roi[0] + center_roi[2], center_roi[1] + center_roi[3]), (0, 255, 0), 2)
        cv2.rectangle(frame, right_roi[:2], (right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]), (255, 0, 0), 2)

        # Run YOLO detection
        results = model(frame, conf=0.4, verbose=False)

        left_objects, center_objects, right_objects = [], [], []

        for r in results:
            for box in r.boxes:
                confidence = round(box.conf[0].item(), 2)
                if confidence > 0.4:
                    obj_name = model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    distance = estimate_distance((x1, y1), (x2, y2))

                    obj_data = f"{obj_name} ({distance}m away)"
                    if x2 <= roi_width:
                        left_objects.append(obj_data)
                    elif x1 >= 2 * roi_width:
                        right_objects.append(obj_data)
                    else:
                        center_objects.append(obj_data)

        # Determine movement direction
        if left_objects and not center_objects and not right_objects:
            dir = f"Move right, object in left: {', '.join(left_objects)}"
        elif right_objects and not center_objects and not left_objects:
            dir = f"Move left, object in right: {', '.join(right_objects)}"
        elif center_objects and not left_objects and not right_objects:
            dir = f"Stop, object in center: {', '.join(center_objects)}"
        elif left_objects and right_objects and not center_objects:
            dir = f"Go straight, objects on both sides: Left - {', '.join(left_objects)}, Right - {', '.join(right_objects)}"
        elif center_objects and (left_objects or right_objects):
            dir = f"Stop, center blocked: {', '.join(center_objects)}"
        else:
            dir = "Path is clear, move forward."

        cv2.imshow('Navigation Mode', frame)

        # Announce direction every 2 seconds
        if time.time() - last_label_time >= 2:
            save_direction(dir)
            print(dir)
            speech.speak(dir)
            last_label_time = time.time()

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speech.speak("Exiting navigation mode.")
            break

    cv2.destroyAllWindows()
