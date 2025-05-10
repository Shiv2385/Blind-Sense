import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150) 

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            user_input = recognizer.recognize_google(audio).lower()
            print("Recognized:", user_input)
            return user_input

    except sr.WaitTimeoutError:
        print("Timeout")
        return None

    except sr.RequestError:
        print("Could not request results from speech recognition service.")
        return None

    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None

    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

