# 🏋️ Exercise Posture Recognition using ML & Deep Learning

## Environment Setup

This project is developed and tested using:

Python 3.10.11

It is recommended to use a virtual environment to avoid dependency conflicts.

### Step 1: Create Virtual Environment

python -m venv post

### Step 2: Activate Environment

Windows:
posture_env\Scripts\activate

### Step 3: Upgrade pip

python -m pip install --upgrade pip

### Step 4: Install Dependencies

pip install numpy==1.23.5
pip install tensorflow==2.10.1
pip install pandas matplotlib seaborn scikit-learn joblib opencv-python mediapipe

Note:
TensorFlow 2.10.1 is compatible with NumPy 1.23.5.
Using newer NumPy versions may cause runtime errors.


## 📌 Overview

This project implements a posture recognition system for multiple exercises using joint-angle based features.

The system compares:

- Classical Machine Learning Models
- 1D Convolutional Neural Network (CNN)
- Hybrid Conv1D + BiLSTM Architecture

The dataset consists of angle-based skeletal features extracted from body joints.

---
Use Python Version 3.10.11 (Important)
## 📂 Dataset Information

- Total Samples: 31,033
- Total Features: 9 Angle-Based Numerical Features
- Classes:
  - Jumping Jacks
  - Pull ups
  - Push Ups
  - Russian twists
  - Squats

Stratified 4-fold cross validation is used for evaluation.

---

## ⚙️ Data Preprocessing

1. Selected only numerical angle features.
2. Encoded class labels using `LabelEncoder`.
3. Standardized features using `StandardScaler`.
4. Reshaped data to 3D format for CNN input.
5. Used stratified splits to maintain class balance.

---

## 🧠 Models Implemented

### 🔹 Classical ML Models

- Random Forest
- SVM (RBF Kernel)
- Logistic Regression
- K-Nearest Neighbors (K=5)
- Decision Tree

Each model is trained on scaled features and evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### 🔹 Deep Learning Models

#### 1️⃣ 1D CNN Architecture

- Conv1D (64 filters)
- MaxPooling
- Conv1D (128 filters)
- Global Max Pooling
- Dense layers
- Softmax output

#### 2️⃣ Hybrid Conv1D + BiLSTM

- Conv1D layers
- MaxPooling
- Bidirectional LSTM
- Dense layers
- Softmax output

---

## 📊 Evaluation Strategy

- 4-Fold Stratified Cross Validation
- Metrics calculated per split
- Final aggregated metrics across all splits
- Confusion matrix visualization

---

## 💾 Model Saving

For each split:

- ML models saved as `.pkl`
- CNN model saved as `.h5`
- Hybrid model saved as `.h5`
- Results stored in `results_store.json`

---

## 🚀 How To Run

Run All
Or run Cell by cell

## Evaluation & Reproducibility Update

To ensure consistent class mapping across training and evaluation, 
the LabelEncoder used during training is saved as:

    label_encoder.pkl

This encoder is loaded during evaluation to maintain identical 
class ordering for confusion matrices and aggregated metrics.

The evaluation is performed using:

    results_store.json

This file stores predictions (`y_true`, `y_pred`) from multiple splits 
for all models:
- ML baselines
- CNN
- Hybrid Conv1D–BiLSTM

Final results are computed by aggregating predictions across all splits.

⚠️ Note:
Do not refit a new LabelEncoder during evaluation. Always reuse 
the saved `label_encoder.pkl` to ensure reproducibility.

🚀 Transformer-Based Posture Classification

This module implements a hybrid Transformer architecture for exercise posture recognition using joint-angle features.

The pipeline is designed to be:

✅ Train-safe (no test leakage)

✅ Test-safe (evaluation only on unseen hold-out data)

✅ Reproducible (saves model, scaler, and label mapping)

🧠 Model Architecture

The model combines:

1D Convolution layers (local feature extraction)

Positional Encoding (sequence awareness)

Multi-Head Self Attention (Transformer block)

GRU layer (temporal refinement)

Fully connected classifier (Softmax output)

