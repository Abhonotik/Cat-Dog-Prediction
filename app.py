import streamlit as st
import pickle
import numpy as np
import cv2
from PIL import Image

# === CONFIG ===
IMG_SIZE = 32
CATEGORIES = ["cat", "dog"]

# === Load Model and Scaler ===
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# === Streamlit UI ===
st.title("🐱🐶 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("L")  # Grayscale
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img_array = np.array(img)
        resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        flat = resized.flatten().reshape(1, -1)
        scaled = scaler.transform(flat)

        # Predict
        prediction = model.predict(scaled)[0]
        label = CATEGORIES[prediction]

        st.success(f"✅ Prediction: **{label.upper()}**")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
