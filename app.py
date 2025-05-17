import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model (converted to .keras)
model = load_model('emotion_model.keras')

# Class labels (must match training order)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def prepare_image(img):
    # Resize and normalize image
    img = img.resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']

    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Unsupported file format'}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img_array = prepare_image(img)

        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            'mood': predicted_label,
            'confidence': round(confidence, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
