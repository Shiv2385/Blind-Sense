import cv2
import os
import numpy as np
import speech

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def check_existing_face(gray_face):
    """Check if the detected face is already saved in the dataset."""
    if not os.path.exists(datasets) or not os.listdir(datasets):
        return None  # No dataset available

    model = cv2.face.LBPHFaceRecognizer_create()
    images, labels, names = [], [], {}

    # ✅ Load dataset correctly (using Code 2 logic)
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

    # ✅ Train model correctly
    model.train(np.array(images, dtype='uint8'), np.array(labels))

    # ✅ Predict if the face exists
    label, confidence = model.predict(gray_face)

    # ✅ Properly check confidence & label validity
    if confidence < 60 and label in names:  # Lower confidence = better match
        return names[label]  # Return correct name
    return None  # Mark as unknown


def save_new_face():
    """Capture and save a new face if not already stored."""
    cap = cv2.VideoCapture(0)
    speech.speak("Looking for a face. Please wait.")

    while True:
        ret, im = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            face_resize = cv2.resize(gray[y:y+h, x:x+w], (130, 100))

            # ✅ Use improved check_existing_face() function
            existing_name = check_existing_face(face_resize)
            if existing_name:
                print(f"This person is {existing_name}, and it is already saved.")
                speech.speak(f"This person is {existing_name}, and it is already saved.")
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit since the face is already stored

            # ✅ If new face, ask for name
            speech.speak("New face detected. Please say the person's name.")
            name = speech.listen()

            if not name:
                speech.speak("Could not recognize the name. Please try again.")
                continue  # Ask again

            path = os.path.join(datasets, name)
            os.makedirs(path, exist_ok=True)

            # ✅ Save 30 face samples
            count = 0
            while count < 30:
                ret, im = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)

                for (x, y, w, h) in faces:
                    face_resize = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                    cv2.imwrite(f'{path}/{count}.png', face_resize)
                    count += 1

                cv2.imshow('Saving Face', im)
                if cv2.waitKey(10) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(f"Face saved for {name}.")
            speech.speak(f"Face saved for {name}.")
            return  # Exit after saving

    cap.release()
    cv2.destroyAllWindows()
