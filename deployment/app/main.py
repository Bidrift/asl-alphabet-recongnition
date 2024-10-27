import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# Set API endpoint
API_URL = "http://localhost:8000/predict"  # Update if needed

st.title("ASL Prediction App")
st.write("Upload an image or take a picture to classify it using the model.")

# Image input options
option = st.selectbox("Choose input method:", ("Upload an image", "Take a picture"))

if option == "Upload an image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
elif option == "Take a picture":
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    # Load the image
    image = Image.open(image_file).convert("L")  # Convert to grayscale
    image = image.resize((32, 32))  # Resize to model input size

    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to 1D list
    image_data = np.array(image).flatten().tolist()

    # Create request payload
    payload = {"image": image_data}

    # Make the prediction request
    if st.button("Predict"):
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            prediction = response.json().get("prediction")
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error in prediction request.")
