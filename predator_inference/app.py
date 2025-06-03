"""
Predator Detection API

This Flask application provides an API for predator detection in images.
It uses a TensorFlow model to classify images into predator categories:
- aguila (eagle)
- mapache (raccoon)
- zorro (fox)
- no_depredador (no predator)

The API exposes endpoints for prediction and image viewing.
"""

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import send_file

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("predator_model.keras")
predator_labels = {0: 'aguila', 1: 'mapache', 2: 'no_depredador', 3: 'zorro'}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predator Detection Endpoint
    
    Receives an image file and returns the predicted predator class with probability.
    
    Request:
        - POST with 'file' field containing an image (jpeg, png, etc.)
    
    Returns:
        - JSON with "prediction" (string) and "probability" (float)
        - HTTP 400 if no file is provided or image decoding fails
        - HTTP 500 for server errors
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"error": "Failed to decode image"}), 400

        img_resized = cv2.resize(img_bgr, (224, 224))

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_array = image.img_to_array(img_rgb)
        img_array = np.expand_dims(img_array, axis=0)
        img_processed = preprocess_input(img_array)

        preds = model.predict(img_processed)[0]
        
        idx = np.argmax(preds)
        prob = preds[idx]
        label = predator_labels.get(idx, "unknown")

        return jsonify({
            "prediction": label,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/view_image')
def view_image():
    """
    Image Viewing Endpoint
    
    Returns the last processed image for debugging purposes.
    
    Request:
        - GET request with no parameters
    
    Returns:
        - JPEG image file
    """
    return send_file("received_image.jpg", mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8501)

