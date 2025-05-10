import object_detection
import faceregtest1
import facesavetest1
import ai_assistant
import navigate
import speech
import cv2
import text_recognition
from object_search import search_object, classNames

def main():
    speech.speak("Give command")

    while True:
        command = speech.listen()

        if command is None:
            continue 

        if "detection" in command:
            speech.speak("Starting object detection")
            object_detection.run_object_detection()

        elif "face" in command:
            speech.speak("Starting face recognition. please be in sufficient light.")
            faceregtest1.run_face_recognition()

        elif "add" in command:
            speech.speak("Saving a new face. please be in sufficient light.")
            facesavetest1.save_new_face()

        elif "chat" in command:
            speech.speak("AI Assistant activated. Say 'Stop AI' to exit.")
            ai_assistant.voice_chatbot()

        elif "read" in command:
            speech.speak("Running Text Recognition...")
            text_recognition.text_recognition()

        elif "search" in command:
            words = command.split()
            if len(words) > 1:
                target_object = " ".join(words[1:])
                if target_object in classNames:
                    speech.speak(f"Searching for {target_object}")
                    search_object(target_object)
                else:
                    speech.speak(f"{target_object} is not a detectable object.")
            else:
                speech.speak("Please specify an object to search.")   

        elif "navigate" in command:
            speech.speak("Starting navigation mode")
            navigate.run_navigation()                 

        elif "exit" in command:
            speech.speak("Exiting the system.")
            break

        else:
            speech.speak("Command not recognized.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()