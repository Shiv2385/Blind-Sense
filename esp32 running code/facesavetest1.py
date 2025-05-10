import cv2
import os
import numpy as np
import speech
import urllib.request  # Added to fetch ESP32 cam image

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# ESP32 Cam Stream URL (Replace with your actual ESP32 cam URL)
ESP32_URL = 'http://192.168.43.142/cam-mid.jpg'  # Change this as needed

def fetch_esp32_frame():
    """Fetch a frame from ESP32 cam stream."""
    try:
        img_resp = urllib.request.urlopen(ESP32_URL)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)  # Decode image
        return frame
    except Exception as e:
        print(f"Error fetching frame: {e}")
        return None

def check_existing_face(gray_face):
    """Check if the detected face is already saved in the dataset."""
    if not os.path.exists(datasets) or not os.listdir(datasets):
        return None  # No dataset available

    model = cv2.face.LBPHFaceRecognizer_create()
    images, labels, names = [], [], {}

    label_id = 0
    for subdir in sorted(os.listdir(datasets)):  # Ensure consistent labeling
        subdir_path = os.path.join(datasets, subdir)
        if not os.path.isdir(subdir_path):
            continue

        names[label_id] = subdir  # Map label ID to name

        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path, 0)
            if img is None:
                continue

            images.append(img)
            labels.append(label_id)

        label_id += 1  # Move to next label

    if not images:
        return None  # No valid images found

    model.train(np.array(images, dtype='uint8'), np.array(labels))
    label, confidence = model.predict(gray_face)

    if confidence < 60 and label in names:
        return names[label]
    return None  # Mark as unknown

def save_new_face():
    """Capture and save a new face if not already stored."""
    speech.speak("Looking for a face. Please wait.")

    while True:
        frame = fetch_esp32_frame()
        if frame is None:
            continue  # Retry fetching frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            face_resize = cv2.resize(gray[y:y+h, x:x+w], (130, 100))

            existing_name = check_existing_face(face_resize)
            if existing_name:
                print(f"This person is {existing_name}, and it is already saved.")
                speech.speak(f"This person is {existing_name}, and it is already saved.")
                return  # Exit since the face is already stored

            speech.speak("New face detected. Please say the person's name.")
            name = speech.listen()

            if not name:
                speech.speak("Could not recognize the name. Please try again.")
                continue  # Ask again

            path = os.path.join(datasets, name)
            os.makedirs(path, exist_ok=True)

            count = 0
            while count < 30:
                frame = fetch_esp32_frame()
                if frame is None:
                    continue  # Retry fetching frame

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)

                for (x, y, w, h) in faces:
                    face_resize = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                    cv2.imwrite(f'{path}/{count}.png', face_resize)
                    count += 1

                cv2.imshow('Saving Face', frame)
                if cv2.waitKey(10) == 27:
                    break

            print(f"Face saved for {name}.")
            speech.speak(f"Face saved for {name}.")
            return  # Exit after saving

    cv2.destroyAllWindows()
