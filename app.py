import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model('emotion_model.h5')

# Class labels (same order as in training)
class_names = ['happy', 'sad', 'angry']  # ⚠️ Match this with your training generator

def prepare_image(img):
    img = img.resize((224, 224)).convert("RGB")  # Ensure RGB format
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'}), 400
    
    file = request.files['image']
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            img = Image.open(file.stream).convert("RGB")
            img_array = prepare_image(img)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_mood = class_names[predicted_index]
            confidence = float(np.max(prediction))

            return jsonify({
                'mood': predicted_mood,
                'confidence': round(confidence * 100, 2)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'File format not supported'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
