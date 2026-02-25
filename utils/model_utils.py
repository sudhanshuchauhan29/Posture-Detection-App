# utils/model_utils.py

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models

def get_ml_models():
    return {
        "RandomForest": RandomForestClassifier(),
        "SVM-RBF": SVC(kernel='rbf'),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier()
    }

def build_cnn(input_shape, n_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])

def build_hybrid(input_shape, n_classes):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(128, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ])