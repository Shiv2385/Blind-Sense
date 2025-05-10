import cv2
import os
import numpy as np
import urllib.request
import speech

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
url = "http://192.168.43.142/cam-mid.jpg"  # ESP32 Camera Stream

def run_face_recognition():
    model = cv2.face.LBPHFaceRecognizer_create()
    names = {}
    images, labels = [], []

    # Load dataset and train model
    if not os.path.exists(datasets):
        print("Dataset folder not found!")
        speech.speak("Dataset folder not found!")
        return

    for subdir in os.listdir(datasets):
        subdir_path = os.path.join(datasets, subdir)
        if not os.path.isdir(subdir_path):
            continue

        names[len(names)] = subdir
        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path, 0)

            if img is None:
                continue  # Skip corrupted images

            images.append(img)
            labels.append(len(names) - 1)

    if not images:
        print("No images found for training!")
        speech.speak("No images found for training!")
        return

    model.train(np.array(images, dtype='uint8'), np.array(labels))

    while True:
        try:
            # Fetch the frame from ESP32 camera
            img_resp = urllib.request.urlopen(url)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(img_np, -1)  # Convert to OpenCV format

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                label, confidence = model.predict(face)

                person_name = names.get(label, "Unknown")
                print(f"{person_name} detected")
                speech.speak(f"{person_name} detected")

                cv2.putText(im, person_name, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('Face Recognition', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                speech.speak("Exiting Face recognition mode.")
                break

        except Exception as e:
            print("Error fetching frame:", str(e))
            speech.speak("Error fetching frame. Retrying...")

    cv2.destroyAllWindows()
