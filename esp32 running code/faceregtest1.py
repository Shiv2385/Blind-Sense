import cv2
import os
import numpy as np
import urllib.request
import speech
import save_new_face  # To run save_new_face.py

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
url = "http://192.168.43.142/cam-mid.jpg"  # ESP32 Camera Stream


def ask_to_save_new_face():
    """Asks the user whether they want to save a new face and runs save_new_face.py if they agree."""
    while True:
        print("No saved dataset found. Do you want to add a new face? Say yes or no.")
        speech.speak("No saved dataset found. Do you want to add a new face? Say yes or no.")
        
        response = speech.listen()
        
        if response is None:
            speech.speak("I didn't understand. Try again.")
            continue

        response = response.lower()

        if "yes" in response:
            speech.speak("Saving a new face.")
            save_new_face.save_new_face()
            return
        elif "no" in response:
            speech.speak("Exiting face recognition mode.")
            return
        else:
            speech.speak("Please give a proper command.")


def ask_to_save_unknown_face():
    """Asks the user whether they want to save a new face and runs save_new_face.py if they agree."""
    while True:
        print("Unknown person detected. Do you want to add a new face? Say yes or no.")
        speech.speak("Unknown person detected. Do you want to add a new face? Say yes or no.")
        
        response = speech.listen()
        
        if response is None:
            speech.speak("I didn't understand. Try again.")
            continue

        response = response.lower()

        if "yes" in response:
            speech.speak("Saving a new face.")
            save_new_face.save_new_face()
            return
        elif "no" in response:
            speech.speak("Exiting face recognition mode.")
            return
        else:
            speech.speak("Please give a proper command.")


def run_face_recognition():
    model = cv2.face.LBPHFaceRecognizer_create()
    names = {}  # Ensure it's always empty at the start
    images, labels = [], []

    # ✅ If the dataset folder does not exist or is empty, ask to save a new face
    if not os.path.exists(datasets) or not os.listdir(datasets):
        ask_to_save_new_face()
        return

    # ✅ Load dataset correctly
    label_id = 0
    for subdir in sorted(os.listdir(datasets)):  # Sorting to ensure consistent label assignment
        subdir_path = os.path.join(datasets, subdir)
        if not os.path.isdir(subdir_path):
            continue

        names[label_id] = subdir  # Assign a unique label to each person

        for filename in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, filename)
            img = cv2.imread(img_path, 0)

            if img is None:
                continue  # Skip corrupted images

            images.append(img)
            labels.append(label_id)  # Ensure correct mapping

        label_id += 1  # Move to the next label

    if not images:  # ✅ If no images are found, ask to add a new face
        ask_to_save_new_face()
        return

    # ✅ Train the model properly
    model.train(np.array(images, dtype='uint8'), np.array(labels))

    while True:
        try:
            img_resp = urllib.request.urlopen(url)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(img_np, -1)

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                label, confidence = model.predict(face)

                # ✅ If confidence is too high or the label is invalid, mark as Unknown
                if confidence > 60 or label not in names:
                    ask_to_save_unknown_face()
                    return
                else:
                    person_name = names[label]  # Get correct name

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
