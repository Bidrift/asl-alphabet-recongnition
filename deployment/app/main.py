import streamlit as st
import requests
import numpy as np
from PIL import Image

API_URL = "http://localhost:8000/predict"

st.title("ASL Prediction App")
st.write("Upload an image or take a picture to classify it.")

option = st.selectbox("Choose input method:", ("Upload an image", "Take a picture"))

if option == "Upload an image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
elif option == "Take a picture":
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    try:
        # Load and preprocess the image
        image = Image.open(image_file).convert('L')  # Convert to grayscale (1 channel)
        image = image.resize((32, 32))  # Resize to 32x32
        image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image_array = (image_array - 0.5) / 0.5  # Normalize using mean=0.5, std=0.5

        st.image(image, caption="Uploaded Image (Preprocessed)", use_column_width=True)

        # Flatten the image for API input
        image_flattened = image_array.flatten().tolist()

        # Create request payload
        payload = {"image": image_flattened}

        # Make the prediction request
        if st.button("Predict"):
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                prediction = response.json().get("prediction")
                st.write(f"Prediction: {prediction}")
            else:
                st.write(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
