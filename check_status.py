import requests
import json

try:
    response = requests.get('http://127.0.0.1:8000/api/training/status')
    if response.status_code == 200:
        print("Backend Status Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed to connect: {e}")
