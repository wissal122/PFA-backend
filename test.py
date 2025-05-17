from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# === Settings ===
model_path = os.path.join(os.path.dirname(__file__), "emotion_model.keras")
# MODEL_PATH = "emotion_model.keras"  # Or use "emotion_model.h5" if needed
IMG_PATH = "test/im0.png"
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Load the model ===
try:
    model = load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# === Prepare the image ===
try:
    img = image.load_img(IMG_PATH, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
except Exception as e:
    print(f"❌ Error loading image: {e}")
    exit()

# === Make prediction ===
try:
    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction))
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(prediction[0][predicted_index]) * 100

    print(f"🎯 Predicted Emotion: {predicted_label} ({confidence:.2f}% confidence)")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
