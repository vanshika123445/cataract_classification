# utils.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
def load_model(model_path="cataract_classifier_model.h5"):
    return tf.keras.models.load_model(model_path)

# Preprocess input image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to the required input shape
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array
