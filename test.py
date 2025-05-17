from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("emotion_model.h5")

# Load image (RGB, resized to match training)
img_path = "test/im0.png"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
pred = model.predict(img_array)

# Define class labels (replace with your actual class names)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # example order

# Get predicted class
predicted_index = np.argmax(pred)
predicted_label = class_names[predicted_index]
confidence = pred[0][predicted_index] * 100

# Output
print(f"Predicted Emotion: {predicted_label} ({confidence:.2f}% confidence)")
