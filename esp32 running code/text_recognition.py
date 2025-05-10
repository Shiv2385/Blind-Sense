import cv2
import easyocr
import urllib.request
import numpy as np
from speech import speak  

# ESP32 Camera Stream URL (Replace with your ESP32 Cam IP)
ESP32_CAM_URL = 'http://192.168.43.142/cam-mid.jpg'  

reader = easyocr.Reader(['en'])  

def text_recognition():
    last_spoken_text = ""  

    while True:
        try:
            # Capture frame from ESP32 Camera
            img_resp = urllib.request.urlopen(ESP32_CAM_URL)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)

            if frame is None or frame.size == 0:
                print("Error: Empty frame received.")
                continue  

            # Perform OCR
            results = reader.readtext(frame)
            detected_text = [res[1] for res in results]

            if detected_text:
                text_output = " | ".join(detected_text)
                print("Detected Text:", text_output)

                if text_output != last_spoken_text:
                    speak(text_output)
                    last_spoken_text = text_output  
            else:
                print("No readable text detected.")

            cv2.imshow('Text Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                speak("Exiting text recognition mode.")
                break

        except Exception as e:
            print(f"Error: {e}")

    cv2.destroyAllWindows()
    print("Text recognition stopped.")
