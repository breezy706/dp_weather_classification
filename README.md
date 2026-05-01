# 🌤️ Weather Prediction using Deep Learning (CNN)

**DATA SCIENTIST: BREEZY_360**

---

## 📌 Project Overview

Weather prediction plays a crucial role in agriculture, transportation, disaster management, and everyday life. Traditional forecasting methods rely on numerical models, satellite data, and complex simulations. However, advancements in Deep Learning now allow accurate weather prediction directly from images.

This project focuses on building a **Convolutional Neural Network (CNN)** model to classify sky images into different weather conditions:
- ☁️ Cloudy  
- 🌧️ Rain  
- ☀️ Shine  
- 🌅 Sunrise  

The model learns visual patterns such as:
- Cloud structures  
- Color intensity  
- Light distribution  

---

## 🎯 Objectives

- Build a CNN-based image classification model  
- Predict weather conditions from sky images  
- Improve accuracy using transfer learning (VGG16)  
- Provide a fast and automated weather prediction system  

---

## 📂 Dataset

👉 **Dataset Link:**  
[https://drive.google.com/drive/folders/1Xg3-spuERoke3zJ9AaT5dtF-be0EnYAZ?usp=drive_link]

---

## 🤖 Trained Model

👉 **Download Model:**  
[https://drive.google.com/file/d/1sIAAgLpHuJBQCiQLkPMKPltwo2I91CT0/view?usp=sharing]

---

## 🔗 GitHub Profile

👉 **My GitHub:**  
https://github.com/Breezy706

---

## 🛠️ Technologies Used

- Python 🐍  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- OpenCV  
- TensorFlow / Keras  

---

## ⚙️ Project Workflow

1. **Data Collection**
   - Multi-class weather image dataset

2. **Data Preprocessing**
   - Image resizing (100x100)
   - Normalization (0–1 scale)
   - Train/Validation/Test split

3. **Model Building (CNN)**
   - Conv2D layers
   - MaxPooling
   - Dropout (to reduce overfitting)
   - Dense layers with Softmax

4. **Training**
   - Loss: `sparse_categorical_crossentropy`
   - Optimizer: `Adam`

5. **Evaluation**
   - Accuracy & Loss metrics
   - Visualization of training vs validation

6. **Prediction**
   - Input image → Model → Weather class output

---

## 📊 Model Performance

| Model Type        | Accuracy |
|------------------|----------|
| Basic CNN        | 79.82%   |
| VGG16 (Pretrained) | **85.09%** |

✅ The pretrained model performed better and is recommended.

---

---
