import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


# Step 1: Rebuild the model architecture
img_width, img_height = 224, 224
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'dataset', 'train')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Step 2: Load only the weights
model.load_weights('emotion_model.h5')

# Step 3: Save in new format
model.save('emotion_model.keras', save_format='keras')
print("✅ Model successfully converted to emotion_model.keras")
