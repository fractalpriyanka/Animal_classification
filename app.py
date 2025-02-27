import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model (make sure the model is saved in your working directory or specify the full path)
model = tf.keras.models.load_model('./my_model.h5')
    
    
# Class labels for your model (example, adjust to your model's labels)
class_labels = {0: 'cat', 1: 'dog', 2: 'snake'}

# Function to preprocess the image and make predictions
def predict_image(img):
    img = img.resize((224, 224))  # Resize image to match model input size (224x224 is common)
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if required by the model

    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class

# Streamlit app interface
st.title('Image Classification with Streamlit')

st.write("Upload an image to classify")

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image.", use_column_width=True)

    # Make prediction
    predicted_class = predict_image(img)
    st.write(f"Predicted Class: {class_labels[predicted_class[0]]}")
