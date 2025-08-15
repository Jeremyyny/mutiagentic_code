# utils.py

import requests

class Textgenwebui:
    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"

    def call_model(self, history):
        headers = {"Content-Type": "application/json"}
        payload = {
            "mode": "chat",
            "messages": history,
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1000
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']


class Ollama:
    def __init__(self, model):
        self.url = "http://localhost:11434/api/chat"
        self.model = model

    def call_model(self, messages):
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False
            }
            response = requests.post(self.url, json=payload, timeout=180)
            response.raise_for_status()
            return response.json().get('message', {}).get('content', '').strip()
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"OLLAMA_API_ERROR: {e}"
