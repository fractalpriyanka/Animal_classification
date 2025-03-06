import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("./my_model.h5")

# Define the known classes (update this with your trained classes)
known_classes = ["cat", "dog", "snakes"]  # Example classes

def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust according to your model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Animal Classification Model")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # Get class label
    predicted_label = known_classes[predicted_class_index] if predicted_class_index < len(known_classes) else "Unknown"

    # If class is unknown, show a warning
    if predicted_label not in known_classes or confidence < 0.6:  # Threshold for confidence
        st.warning("Hey I'm not capable of recognizing this !")
    else:
        st.header(f"Predicted class: {predicted_label}")
