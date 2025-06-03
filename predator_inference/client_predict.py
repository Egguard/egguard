# client.py
import requests

IMAGE_PATH = "sillas.jpg"
ENDPOINT_URL = "http://localhost:8501/predict"

with open(IMAGE_PATH, "rb") as f:
    files = {"file": f}
    response = requests.post(ENDPOINT_URL, files=files)

if response.status_code == 200:
    result = response.json()
    print("✅ Prediction:", result["prediction"])
    print("📊 Probability:", result["probability"])
else:
    print("❌ Error:", response.status_code, response.text)
