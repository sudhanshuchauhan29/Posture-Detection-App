
# Environment Configuration

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are saving your model as an HDF5 file"
)

# Standard Libraries

import json
import joblib
import numpy as np
import pandas as pd

# Visualization 


import matplotlib.pyplot as plt
import seaborn as sns


# Machine Learning Models

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ==============================
# Deep Learning
# ==============================

import tensorflow as tf
from tensorflow.keras import layers, models

# ==============================
# Model & Results Loading
# ==============================

from tensorflow.keras.models import load_model


def load_saved_results():
    """Load stored experiment results if available."""
    if not os.path.exists("results_store.json"):
        return None

    try:
        with open("results_store.json", "r") as f:
            data = json.load(f)
        print("✔ Previous results_store loaded.")
        return data
    except Exception as e:
        print(f"✘ Error loading results_store.json: {e}")
        return None


def load_saved_models(split_id):
    """Load saved ML and DL models for given split."""
    models_dict = {}

    # Classical ML models
    ml_names = [
        "RandomForest",
        "SVM-RBF",
        "LogisticRegression",
        "KNN-5",
        "DecisionTree"
    ]

    for name in ml_names:
        path = f"ml_model_split{split_id}_{name}.pkl"
        if os.path.exists(path):
            models_dict[name] = joblib.load(path)
            print(f"✔ Loaded {name} model")

    # CNN model
    cnn_path = f"cnn_split{split_id}.h5"
    if os.path.exists(cnn_path):
        models_dict["CNN"] = load_model(cnn_path)
        print("✔ Loaded CNN model")

    # Hybrid model
    hybrid_path = f"hybrid_split{split_id}.h5"
    if os.path.exists(hybrid_path):
        models_dict["Hybrid"] = load_model(hybrid_path)
        print("✔ Loaded Hybrid model")

    return models_dict

# ==============================
# Main Execution
# ==============================

def main():
    results_store = load_saved_results()

    if results_store:
        print("\n🎉 All saved results found. Training is NOT required.")
    else:
        print("\n⚠ No saved results found. Please run training function.")


if __name__ == "__main__":
    main()

df = pd.read_csv("exercise_angles.csv")
print("Dataset Loaded Successfully!\n")

# ============================================================
# 📌 CREATE 4 STRATIFIED SPLITS OF THE DATASET (LODO SETUP)
# ============================================================

from sklearn.model_selection import StratifiedKFold

# Separate features + labels
X_full = df.select_dtypes(include=['float64','int64']).iloc[:, :-1].values
y_full = df["Label"].values

# Create 4 stratified folds
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

splits = []  # This will store the 4 dataset partitions

for train_idx, test_idx in skf.split(X_full, y_full):
    X_train_split = X_full[train_idx]
    y_train_split = y_full[train_idx]
    X_test_split = X_full[test_idx]
    y_test_split = y_full[test_idx]
    splits.append((X_train_split, y_train_split, X_test_split, y_test_split))

print("\nDataset successfully split into 4 stratified parts!")

# ============================================================
# 📌 PRINT DETAILS OF ALL 4 SPLITS
# ============================================================
for i, (X_train_split, y_train_split, X_test_split, y_test_split) in enumerate(splits):
    print(f"\n========== SPLIT {i+1} ==========")
    print(f"Training samples: {len(X_train_split)}")
    print(f"Testing samples : {len(X_test_split)}")
    print(f"Training class distribution:")
    
    # Class count in train
    unique, counts = np.unique(y_train_split, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label}: {count}")
    
    print(f"Testing class distribution:")
    
    # Class count in test
    unique, counts = np.unique(y_test_split, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"   {label}: {count}")

print("🔹 First 5 Rows:\n", df.head(), "\n")
print("🔹 Shape:", df.shape, "\n") # Tell the number of rows and column
print("🔹 Columns:\n", df.columns, "\n") 
print("🔹 Dataset Info:\n")
print(df.info())
print()
print("🔹 Missing Values:\n", df.isnull().sum(), "\n")
plt.figure(figsize=(6,4))
df["Label"].value_counts().plot(kind="bar")
plt.title("Label Distribution")
plt.xlabel("Posture / Exercise Type")
plt.ylabel("Count")
plt.show()
print("🔹 Statistical Summary of Numerical Columns:\n")
print(df.describe(), "\n")
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Between Angles")
plt.show()
print("🔹 Unique Classes:", df["Label"].unique())
import numpy as np 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
import joblib
# Detect numeric columns only
numeric_df = df.select_dtypes(include=['float64', 'int64']) # isolate numeric column
# Labels
y = df.iloc[:, -1].values # labbel
X = numeric_df.iloc[:, :-1].values # all numeric values
# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
np.save("label_classes.npy", le.classes_) # save ki file
# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# SAVE scaler
joblib.dump(scaler, "angle_scaler.pkl")
# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)
#CNN ko  3d input chahiye uske liye

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Final X shape:", X.shape)
print("Classes:", le.classes_)
# ============================================================
# 📌 Utility Functions for Confusion Matrix, Accuracy, Loss
# ============================================================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import numpy as np

def plot_confusion_matrix_custom(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

def plot_training_curves(history, model_name, split_id):
    # Accuracy Plot
    plt.figure(figsize=(6,4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"{model_name} Accuracy Curve (Split {split_id})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

import os
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import models, layers

results_store = {}

os.makedirs("models/ml", exist_ok=True)
os.makedirs("models/cnn", exist_ok=True)
os.makedirs("models/hybrid", exist_ok=True)

for split_id in range(4):

    print(f"\n===== TRAINING SPLIT {split_id+1} =====")

    X_test = splits[split_id][2]
    y_test = le.transform(splits[split_id][3])

    X_train = np.vstack([splits[j][0] for j in range(4) if j != split_id])
    y_train = np.hstack([splits[j][1] for j in range(4) if j != split_id])
    y_train = le.transform(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results_store[split_id+1] = {
        "ML": {},
        "CNN": {},
        "Hybrid": {}
    }

    # ================= ML MODELS =================
    ml_models = {
        "RandomForest": RandomForestClassifier(),
        "SVM-RBF": SVC(kernel='rbf'),
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier()
    }

    for name, clf in ml_models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)

        results_store[split_id+1]["ML"][name] = {
            "y_true": y_test.tolist(),
            "y_pred": preds.tolist()
        }

        joblib.dump(clf, f"models/ml/split{split_id+1}_{name}.pkl")

    # ================= CNN =================
    X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
    X_test_cnn  = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

    cnn = models.Sequential([
        layers.Input(shape=(X_train_cnn.shape[1], 1)),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(le.classes_), activation='softmax')
    ])

    cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    cnn.fit(X_train_cnn, y_train,
            validation_data=(X_test_cnn, y_test),
            epochs=15,
            batch_size=32,
            verbose=0)

    y_pred_cnn = np.argmax(cnn.predict(X_test_cnn), axis=1)

    results_store[split_id+1]["CNN"] = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred_cnn.tolist()
    }

    cnn.save(f"models/cnn/split{split_id+1}.h5")

    # ================= HYBRID =================
    hybrid = models.Sequential([
        layers.Input(shape=(X_train_cnn.shape[1], 1)),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(le.classes_), activation='softmax')
    ])

    hybrid.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    hybrid.fit(X_train_cnn, y_train,
               validation_data=(X_test_cnn, y_test),
               epochs=15,
               batch_size=32,
               verbose=0)

    y_pred_hybrid = np.argmax(hybrid.predict(X_test_cnn), axis=1)

    results_store[split_id+1]["Hybrid"] = {
        "y_true": y_test.tolist(),
        "y_pred": y_pred_hybrid.tolist()
    }

    hybrid.save(f"models/hybrid/split{split_id+1}.h5")

# Save results
with open("results_store.json", "w") as f:
    json.dump(results_store, f)

print("🎉 Training complete. All models saved.")



from sklearn.preprocessing import LabelEncoder
import joblib

# y = your original label column (before encoding)
# Example:
# y = df["label"]

le = LabelEncoder()
le.fit(y)

joblib.dump(le, "label_encoder.pkl")

print("✔ LabelEncoder saved successfully")
print("Classes:", le.classes_)

import os
import json
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load encoder
le = joblib.load("label_encoder.pkl")

# Load results
with open("results_store.json", "r") as f:
    results_store = json.load(f)

results_store = {int(k): v for k, v in results_store.items()}

def final_aggregated_metrics(results_store, le, model_type, model_name=None):

    all_y_true, all_y_pred = [], []

    for split_id in sorted(results_store.keys()):
        if model_type == "ML":
            data = results_store[split_id]["ML"][model_name]
        else:
            data = results_store[split_id][model_type]

        all_y_true.extend(data["y_true"])
        all_y_pred.extend(data["y_pred"])

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    f1   = f1_score(y_true, y_pred, average="weighted")

    print("\n================ FINAL RESULT ================")
    print("Model     :", model_name if model_name else model_type)
    print("Accuracy  :", round(acc, 4))
    print("Precision :", round(prec, 4))
    print("Recall    :", round(rec, 4))
    print("F1-score  :", round(f1, 4))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title(f"Final Confusion Matrix – {model_name if model_name else model_type}")
    plt.show()

# 🔥 Run for all models
final_aggregated_metrics(results_store, le, "Hybrid")
final_aggregated_metrics(results_store, le, "CNN")
final_aggregated_metrics(results_store, le, "ML", "RandomForest")
final_aggregated_metrics(results_store, le, "ML", "SVM-RBF")
final_aggregated_metrics(results_store, le, "ML", "LogisticRegression")
final_aggregated_metrics(results_store, le, "ML", "KNN-5")
final_aggregated_metrics(results_store, le, "ML", "DecisionTree")