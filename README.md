# 🧠 Age-group Classification using VGG16 (Transfer Learning)

This project classifies human face images into predefined **age groups** using **deep learning** and **transfer learning** with a pre-trained **VGG16** model.  
It demonstrates end-to-end workflow — from data preprocessing and model training to fine-tuning, evaluation, and model export.

---

## 📘 Project Overview
- **Objective:** Predict the age group of a person based on their facial image.
- **Approach:** Transfer Learning using the **VGG16** convolutional neural network pretrained on ImageNet.
- **Framework:** TensorFlow / Keras
- **Output:** A trained model (`age_classification_final_vgg16.keras`) capable of classifying faces into multiple age groups.

---

## ⚙️ Model Architecture
- **Base Model:** VGG16 (pre-trained on ImageNet,top convolutional layers ('block-5') unfreezed)
- **Custom Head:**
  - Global Average Pooling layer
  - Dense layer(s) with ReLU activation
  - Dropout layer to prevent overfitting
  - Final Dense layer with Softmax activation for multi-class classification
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy

---

## 🧩 Training Strategy
- **Data Augmentation:** Rotation, flipping, zooming, brightness and shift augmentations.
- **Fine-Tuning:** Unfrozen top convolutional layers for better feature learning.
- **Regularization Techniques:** Dropout, early stopping, and model checkpointing.
- **Batch Size / Epochs:** Tuned experimentally for optimal accuracy.

---

## 📊 Results
| Metric | Value |
|--------|-------|
| **Training Accuracy** | ~<82>% |
| **Validation Accuracy** | ~<77>% |
| **Loss Function** | Categorical Crossentropy |
| **Best Epoch** | <29> |


---

## 🧠 Dataset
- **Dataset:** `<UTKFace>` 
- **Number of Images:** `<23.7K>`  
- **Number of Classes:** `<5>` age groups ( 0–10, 11–29, 20–35, 36-60, 60+)
- **Preprocessing:**
  - Resized images to `(224, 224, 3)`
  - Normalized pixel values to `[0, 1]`
  - Split into training, validation (80/20)

---

## 🚀 Usage

### 🔹 1. Setup Environment
```bash
git clone https://github.com/<your-username>/Age-Classification-VGG16.git
cd Age-Classification-VGG16
pip install -r requirements.txt
