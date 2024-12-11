import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model function with caching for faster reloads
@st.cache_resource
def load_trained_model():
    model_path = 'lulc100.h5'  # Ensure this path is correct
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        return None
    return load_model(model_path)

# Load the trained model
model = load_trained_model()

# Define class names
index_to_class = {
    0: 'Annual Crop',
    1: 'Forest',
    2: 'Herbaceous Vegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'Permanent Crop',
    7: 'Residential',
    8: 'River',
    9: 'Sea Lake',
}

# Function to preprocess image
def load_and_preprocess_image(img, target_size=(64, 64)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI

st.title("🌍 Environmental Classification AI")
st.markdown(
    """
    <style>
        .stApp {
            background-image: url("https://drive.google.com/file/d/1Z-bHwaTQG8ni2Os5UYqjTnfkjvtqbTM2/view?usp=drive_link");
            background-size: cover;
            background-position: center;
        }
        .main-title { font-size: 2rem; font-weight: bold; color: #4CAF50; }
        .prediction { font-size: 1.5rem; font-weight: bold; color: #FF5722; }
        .confidence { font-size: 1.2rem; color: #009688; }
        .title {font-size: 10rem; font-weight: bold; color: #4CAF50; }
    </style>
    """, unsafe_allow_html=True
)
uploaded_file = st.file_uploader("Upload a satellite image to classify", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = load_and_preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = predictions[0][predicted_class_index] * 100
    predicted_class_name = index_to_class.get(predicted_class_index, "Unknown")

    # Display prediction results
    st.markdown(f"<div class='prediction'>Predicted Class: {predicted_class_name}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence'>Confidence: {confidence_score:.2f}%</div>", unsafe_allow_html=True)

# Footer for extra styling
st.markdown("---")
st.markdown("Developed by [Team - S7] 🚀", unsafe_allow_html=True)
