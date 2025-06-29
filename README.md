# 🐱🐶 Cat vs Dog Image Classifier

A machine learning web app that classifies images as **Cat** or **Dog** using a **Support Vector Machine (SVM)** model and deployed using **Streamlit**.

---

## 🚀 Features

- Upload an image and get instant prediction: **Cat** or **Dog**
- Uses grayscale images resized to 32x32 for efficient training
- Model trained using **SVM with linear kernel**
- **StandardScaler** used for preprocessing
- Web app built with **Streamlit**
- Deployed via **Render**

---

## 📁 Project Structure

Dog_Cat_Prediction/
├── app.py # Streamlit frontend
├── svm_model.pkl # Trained SVM model
├── scaler.pkl # StandardScaler for image features
├── requirements.txt # Python dependencies
├── train/ # Training images (cats & dogs)
├── test/ # Test images (optional)
└── predict_image/ # Folder to test single images

yaml
Copy
Edit

---

## 🧠 Model Training Summary

- Dataset: [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
- Preprocessing:
  - Grayscale conversion
  - Resize to 32x32
  - Flatten to 1D vector
  - Standardization
- Model: `SVC(kernel='linear')`
- Evaluation:
  - Accuracy score
  - Classification report

---

## 🌐 Running Locally

1. **Clone the repo:**

```bash
git clone https://github.com/Abhonotik/Cat-Dog-Prediction.git
cd Cat-Dog-Prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
🧪 Testing an Image
You can test a new image by placing it in the predict_image/ folder and selecting it from the Streamlit interface.

🛠 Built With
Python

OpenCV

NumPy

scikit-learn

Streamlit

