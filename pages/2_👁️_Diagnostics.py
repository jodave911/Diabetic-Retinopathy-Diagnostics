import os
import sys
from contextlib import contextmanager

import tensorflow as tf
import streamlit as st
import cv2
import numpy as np

# Define context manager to suppress stdout
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Load the Keras model
DR_model = tf.keras.models.load_model("DR_CNN_Model.keras")

# Preprocess image function
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.array(img) / 255.0
    return img

# Function to predict class
def predict_class(img):
    # Suppress printing during model prediction
    with suppress_stdout():
        # Preprocess the image
        image_arr = preprocess_image(img)

        # Make predictions
        prediction = DR_model.predict(np.array([image_arr]))

    # Post-process predictions
    predicted = np.argmax(prediction, axis=1)
    confidence = np.max(prediction) * 100

    # Return the predicted class and confidence
    if predicted == 1:
        return 'DR', confidence
    else:
        return 'NO_DR', confidence

# Main function
def main():
    st.set_page_config(page_title="Diabetic Retinopathy Diagnostics")
    st.title("Diabetic Retinopathy Classification")
    st.write("Upload an image to classify whether it has diabetic retinopathy or not.")
    st.write("Supported formats: JPG, JPEG, PNG")
    st.write("Note: For best results, upload an image with clear retinal details.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner('Classifying...'):
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        st.image(img, caption='Uploaded Image.', use_column_width=True)

        class_prediction, confidence = predict_class(img)
        st.write(f"Prediction: **{class_prediction}** with **{confidence:.2f}%** confidence.")

if __name__ == "__main__":
    main()
