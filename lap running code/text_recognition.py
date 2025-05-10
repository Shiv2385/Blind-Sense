import cv2
import easyocr
from speech import speak  

reader = easyocr.Reader(['en'])  

def text_recognition():
    camera = cv2.VideoCapture(0)
    last_spoken_text = ""  

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Camera feed not available.")
            break

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

    camera.release()
    cv2.destroyAllWindows()
    print("Text recognition stopped.")