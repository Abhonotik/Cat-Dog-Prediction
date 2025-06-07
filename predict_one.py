import pickle
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = 32
CATEGORIES = ["cat", "dog"]
image_path = "predict_image/cat.1005.jpg"  # <- Change this to your test image path

# === LOAD MODEL AND SCALER ===
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# === LOAD AND PREPROCESS IMAGE ===
try:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable.")

    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    flat = resized.flatten().reshape(1, -1)
    scaled = scaler.transform(flat)

    # === PREDICT ===
    prediction = model.predict(scaled)[0]
    label = CATEGORIES[prediction]

    print(f"✅ Prediction: {label}")

    # === SHOW IMAGE ===
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()

except Exception as e:
    print("❌ Error:", e)
