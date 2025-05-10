import cv2
import os
import speech
import urllib.request
import numpy as np

datasets = 'datasets'
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# ESP32 Camera Stream URL (Replace with your ESP32 Cam IP)
ESP32_CAM_URL = 'http://192.168.43.142/cam-mid.jpg'

def save_new_face():
    speech.speak("Say the person's name.")
    name = speech.listen()

    if not name:  # If speech recognition fails
        speech.speak("Could not recognize the name. Please try again.")
        return

    path = os.path.join(datasets, name)
    os.makedirs(path, exist_ok=True)

    count = 0
    while count < 30:
        try:
            # Capture frame from ESP32 Camera
            img_resp = urllib.request.urlopen(ESP32_CAM_URL)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            im = cv2.imdecode(imgnp, -1)

            if im is None or im.size == 0:
                print("Error: Empty frame received.")
                continue  

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            for (x, y, w, h) in faces:
                face_resize = cv2.resize(gray[y:y+h, x:x+w], (130, 100))
                cv2.imwrite(f'{path}/{count}.png', face_resize)
                count += 1

            cv2.imshow('Saving Face', im)
            if cv2.waitKey(10) == 27:
                break

        except Exception as e:
            print(f"Error: {e}")

    cv2.destroyAllWindows()
    print(f"Face saved for {name}.")
    speech.speak(f"Face saved for {name}.")
