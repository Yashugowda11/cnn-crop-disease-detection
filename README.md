# 🌱 CNN Crop Disease Detection

A deep learning-based web application that detects crop diseases from leaf images using Convolutional Neural Networks (CNN).

## 📌 Project Overview

The system allows users to upload an image of a crop leaf and predicts the possible disease using a trained deep learning model.

### Main Features

- 🌿 Upload crop leaf images
- 🤖 CNN-based disease detection
- 📊 Disease prediction with confidence
- 📈 Severity indication
- 📄 Generate/download prediction reports
- 🌐 Web-based user interface

## 🛠️ Technologies Used

### Frontend
- React
- Tailwind CSS

### Backend
- Python
- Flask
- REST API

### Machine Learning
- TensorFlow
- Keras
- CNN
- MobileNetV2
- Transfer Learning

### Dataset
- PlantVillage Dataset
- Data augmentation
- 14 crop types
- 38 disease classes

## 🧠 Machine Learning Model

The project uses **MobileNetV2 with Transfer Learning** for crop disease classification.

The input leaf image is preprocessed and passed through the trained model. The model then predicts the disease class and provides the prediction confidence.

## 🔄 System Workflow

```text
Crop Leaf Image
       ↓
Image Upload
       ↓
Image Preprocessing
       ↓
MobileNetV2 / CNN Model
       ↓
Disease Classification
       ↓
Prediction & Confidence
       ↓
Severity Result
       ↓
Report

cnn-crop-disease-detection/
│
├── backend/
│
├── frontend/
│
├── dataset/
│   └── dataset/
│       └── plantvillage dataset/
│
├── .gitattributes
├── .gitignore
└── README.md
