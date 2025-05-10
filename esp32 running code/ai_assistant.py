import requests
import json
import re
import speech  # ✅ Using existing speech module

# API Key (Keep it private)
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def modify_input(user_message):
    """Modify input for structured response."""
    return user_message + " (Provide a clear and concise response as I am blind. No emoji needed.)"

def clean_response(text):
    """Clean unwanted characters from response."""
    text = re.sub(r'[*#_`]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def chat_with_model(user_message):
    modified_message = modify_input(user_message)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-r1-distill-llama-70b:free",
        "messages": [{"role": "user", "content": modified_message}],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        raw_text = response.json()

        if "choices" in raw_text and len(raw_text["choices"]) > 0:
            return clean_response(raw_text["choices"][0]["message"]["content"])
        else:
            return "Error: No response received from AI."
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except (KeyError, IndexError, json.JSONDecodeError):
        return "Error: Unexpected response format."

def voice_chatbot():

    while True:
        user_input = speech.listen() 

        if user_input:
            if "stop ai" in user_input:
                speech.speak("Exiting AI Assistant mode.")
                break

            print(f"You: {user_input}")
            bot_reply = chat_with_model(user_input)
            print(f"Chatbot: {bot_reply}")
            speech.speak(bot_reply)
