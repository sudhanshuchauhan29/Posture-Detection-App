🏋️ Exercise Posture Recognition using ML & Deep Learning

Real-time exercise posture recognition system using joint-angle features and a Sequence-Based Hybrid Conv1D–BiLSTM architecture.

This project compares classical machine learning, CNN, Transformer, and a proposed hybrid deep learning model for accurate exercise classification.

🔍 Overview

This system:

Extracts 33 human pose landmarks using MediaPipe

Converts landmarks into biomechanically meaningful joint angles

Forms 20-frame temporal sequences

Uses a Hybrid Conv1D + BiLSTM model

Performs real-time posture classification

Unlike frame-based approaches, this model captures motion continuity, improving robustness and accuracy.

🧠 Models Implemented
🔹 Classical ML Models

Random Forest

SVM (RBF)

Logistic Regression

KNN (k=5)

Decision Tree

🔹 Deep Learning Models
1️⃣ 1D CNN (Single Frame)

Frame-based angle classification.

2️⃣ Hybrid Conv1D–BiLSTM (Single Frame)

Conv1D + BiLSTM without temporal sequence modeling.

3️⃣ Transformer-Based Sequence Model

Conv1D + Positional Encoding + Multi-Head Attention + GRU.

4️⃣ 🚀 Proposed Model: Sequence-Based Hybrid Conv1D–BiLSTM

Operates on 20 consecutive frames of joint-angle data.

Architecture:

Input (20 × 10)
        ↓
Conv1D (64)
        ↓
Conv1D (128)
        ↓
MaxPooling
        ↓
Bidirectional LSTM (128)
        ↓
Dense (128) + Dropout
        ↓
Softmax Output
📊 Results
Model	Accuracy
CNN (Single Frame)	93.09%
Hybrid (1 Frame)	96.09%
Transformer (Sequence)	91.09%
Hybrid (20 Seq)	97.85%

✔ Temporal modeling significantly improves performance.
✔ Hybrid Conv1D–BiLSTM achieves the best overall results.
✔ Transformer underperforms for short structured angle sequences.

📂 Dataset Information

Total Samples: 31,033

Features: 9–10 Joint-Angle Numerical Features

Classes:

Jumping Jacks

Pull ups

Push Ups

Russian twists

Squats

Data is generated from pose estimation and converted into angle-based representations.

⚙️ Preprocessing Pipeline

Extract pose landmarks (MediaPipe)

Compute joint angles

Encode labels (LabelEncoder)

Normalize features (StandardScaler)

Create rolling sequence buffer (20 frames)

Train hybrid model

🚀 Real-Time Inference Pipeline
Video Capture (OpenCV)
        ↓
Pose Estimation (MediaPipe)
        ↓
Joint Angle Calculation
        ↓
Feature Normalization
        ↓
Sequence Buffer (20 Frames)
        ↓
Hybrid Model Prediction
        ↓
Exercise Classification Output

Designed for low-latency real-time deployment.

💾 Saved Artifacts

sequence_hybrid_model.h5 → Trained deep model

sequence_scaler.pkl → StandardScaler

sequence_labels.npy → Class mapping

sequence_history.json → Training + evaluation metrics

results_store.json → Cross-validation results

label_encoder.pkl → Reproducible label encoding

🖥️ Environment Setup
🔹 Required Version
Python 3.10.11

TensorFlow 2.10.1 requires NumPy 1.23.5.

🔹 Installation
python -m venv post
post\Scripts\activate
python -m pip install --upgrade pip

pip install numpy==1.23.5
pip install tensorflow==2.10.1
pip install pandas matplotlib seaborn scikit-learn joblib opencv-python mediapipe
▶️ How To Run

Place:

exercise_angles.csv

in the project directory.

Then run:

python sequence_model.py

If saved model exists, training will be skipped automatically.

📈 Key Contributions

Joint-angle based compact representation

Sequence-aware posture modeling

Hybrid Conv1D + BiLSTM architecture

Comparative evaluation with ML, CNN, and Transformer

Real-time compatible pipeline

EarlyStopping with best-weight restoration

🔮 Future Improvements

Multi-person detection support

More exercise categories

Adaptive sequence length

Lightweight attention integration

Posture correctness scoring

Edge-device deployment

🎯 Applications

Fitness monitoring

Sports coaching systems

Rehabilitation tracking

Smart gym analytics

Home-based workout monitoring

📄 Academic Documentation

Full academic report available in repository.

👨‍💻 Author

Sudhanshu Chauhan
B.Tech Computer Science & Engineering