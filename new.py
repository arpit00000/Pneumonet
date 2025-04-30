import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import os

# Load your trained model
model = load_model(r"C:\Users\HP\Downloads\c320\320\Lung-Disease-Detection-using-CNN\RESNET-101.keras")  # Update with your correct path

# Class labels
class_labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Set up the Streamlit app
st.title("Medical Image Classification")
st.write("Upload a chest X-ray image for classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Load and preprocess the image
        img = load_img("temp_image.jpg", target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Display the image with prediction
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Show prediction results
        st.success(f"Predicted: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Show probabilities for all classes
        st.subheader("Class Probabilities:")
        for i, class_label in enumerate(class_labels):
            st.write(f"{class_label}: {predictions[0][i]*100:.2f}%")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists("temp_image.jpg"):
            os.remove("temp_image.jpg")