import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_chat():
    print("Testing /api/chat...")
    
    payload = {
        "question": "Quelle est l'estimation du capital requis ?"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Answer:", data.get("answer")[:100] + "...")
            print("Sources present:", "sources" in data)
            print("Metrics present:", "metrics" in data)
        else:
            print("Chat failed:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_chat()
