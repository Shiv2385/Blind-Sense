import cv2
import os
import speech

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def save_new_face():
    cap = cv2.VideoCapture(0)
    speech.speak("Say the person's name.")

    name = speech.listen()

    if not name:  # If speech recognition fails
        speech.speak("Could not recognize the name. Please try again.")
        cap.release()
        return

    path = os.path.join(datasets, name)
    os.makedirs(path, exist_ok=True)

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
