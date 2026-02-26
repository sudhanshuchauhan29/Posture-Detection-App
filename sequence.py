# ================================================================
# SEQUENCE HYBRID CONV1D + BiLSTM MODEL
# Train Once → Save → Always Load
# ================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns


# ================== PARAMETERS ==================
CSV_PATH = "exercise_angles.csv"
SEQ_LEN = 20

MODEL_PATH = "sequence_hybrid_model.h5"
SCALER_PATH = "sequence_scaler.pkl"
LABELS_PATH = "sequence_labels.npy"
HISTORY_PATH = "sequence_history.json"


# ================================================================
# 1️⃣ LOAD MODEL IF EXISTS
# ================================================================
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(LABELS_PATH):
    print("✔ Model Found — Loading...")

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)

        print("\n📊 Saved Metrics:")
        print("Train Accuracy :", history["final_train_accuracy"])
        print("Val Accuracy   :", history["final_val_accuracy"])
        print("Test Accuracy  :", history["test_accuracy"])
        print("Test Precision :", history["test_precision"])
        print("Test Recall    :", history["test_recall"])
        print("Test F1 Score  :", history["test_f1"])

    print("\n✅ Model Ready for Inference")
    exit()


# ================================================================
# 2️⃣ TRAINING MODE
# ================================================================
print("⚠ Training Model...")

df = pd.read_csv(CSV_PATH)

# -------- Feature Columns --------
possible_angle_cols = [
    'Shoulder_Angle','Elbow_Angle','Hip_Angle','Knee_Angle','Ankle_Angle',
    'Shoulder_Ground_Angle','Elbow_Ground_Angle','Hip_Ground_Angle',
    'Knee_Ground_Angle','Ankle_Ground_Angle'
]

angle_cols = [c for c in possible_angle_cols if c in df.columns]
if not angle_cols:
    angle_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()

# -------- Label Column --------
label_col = None
for c in ["label_enc", "Label", "label", "label_encoded"]:
    if c in df.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError("No label column found")

# -------- Build Sequences --------
X_raw = df[angle_cols].values.astype(np.float32)
y_raw = df[label_col].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

seq_X, seq_y = [], []
buffer = deque(maxlen=SEQ_LEN)

for i in range(len(X_raw)):
    buffer.append(X_raw[i])
    if len(buffer) == SEQ_LEN:
        seq_X.append(np.array(buffer))
        seq_y.append(y[i])

seq_X = np.array(seq_X)
seq_y = np.array(seq_y)

print("Sequence Shape:", seq_X.shape)

# -------- Scaling --------
n_samples, seq_len, n_feats = seq_X.shape
scaler = StandardScaler()
seq_X = scaler.fit_transform(seq_X.reshape(-1, n_feats)).reshape(n_samples, seq_len, n_feats)

joblib.dump(scaler, SCALER_PATH)
np.save(LABELS_PATH, le.classes_)

# -------- Train-Test Split --------
X_train, X_test, y_train, y_test = train_test_split(
    seq_X, seq_y, test_size=0.2, random_state=42, stratify=seq_y
)


# ================================================================
# 3️⃣ BUILD MODEL
# ================================================================
def build_model(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(128))(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inp, out)


model = build_model((SEQ_LEN, n_feats), len(le.classes_))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=3),
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
]


# -------- Train --------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)


# ================================================================
# 4️⃣ EVALUATION
# ================================================================
final_train_acc = history.history["accuracy"][-1]
final_val_acc   = history.history["val_accuracy"][-1]

y_pred = np.argmax(model.predict(X_test), axis=1)

test_acc  = accuracy_score(y_test, y_pred)
test_prec = precision_score(y_test, y_pred, average="weighted")
test_rec  = recall_score(y_test, y_pred, average="weighted")
test_f1   = f1_score(y_test, y_pred, average="weighted")

print("\n📊 Final Results:")
print("Train Accuracy :", final_train_acc)
print("Val Accuracy   :", final_val_acc)
print("Test Accuracy  :", test_acc)
print("Test Precision :", test_prec)
print("Test Recall    :", test_rec)
print("Test F1 Score  :", test_f1)


# ================================================================
# 5️⃣ SAVE MODEL + HISTORY
# ================================================================
model.save(MODEL_PATH)

history_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
history_clean.update({
    "final_train_accuracy": float(final_train_acc),
    "final_val_accuracy": float(final_val_acc),
    "test_accuracy": float(test_acc),
    "test_precision": float(test_prec),
    "test_recall": float(test_rec),
    "test_f1": float(test_f1)
})

with open(HISTORY_PATH, "w") as f:
    json.dump(history_clean, f)

print("\n🎉 Model Saved Successfully!")


# ================================================================
# 6️⃣ PLOTS + CONFUSION MATRIX
# ================================================================
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.show()

print("\n✅ DONE — Sequence Hybrid Model Trained, Evaluated & Saved!")