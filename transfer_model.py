# ============================================================
# TRANSFORMER-BASED POSTURE CLASSIFICATION
# Train-Safe + Test-Safe Implementation
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH   = "data.csv"  # <-- change to your dataset file
MODEL_PATH  = "transformer_final.h5"
SCALER_PATH = "transformer_scaler.pkl"
LABEL_PATH  = "transformer_labels.npy"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
        angles = pos * angle_rates

        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(angles[:, 0::2])
        pe[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.cast(pe[None, ...], tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

def transformer_block(x, num_heads=2, key_dim=32, ff_dim=128):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(x, x)

    x = layers.LayerNormalization()(x + attn)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(key_dim * num_heads)(ff)
    x = layers.LayerNormalization()(x + ff)

    return x


# ============================================================
# BUILD MODEL
# ============================================================

def build_transformer(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = PositionalEncoding(x.shape[1], 64)(x)
    x = transformer_block(x)

    x = layers.GRU(48)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(n_classes, activation="softmax")(x)

    return models.Model(inp, out)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_angles(X, noise_std=0.8, n_copies=2):
    return np.vstack([
        X + np.random.normal(0, noise_std, X.shape)
        for _ in range(n_copies)
    ])


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df.select_dtypes(include=['float64', 'int64']).iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # --------------------------------------------------------
    # LABEL ENCODING
    # --------------------------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    label_classes = list(le.classes_)
    n_classes = len(label_classes)

    # --------------------------------------------------------
    # TRAIN-TEST SPLIT
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_enc
    )

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # --------------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------------
    X_aug = augment_angles(X_train_scaled)
    y_aug = np.tile(y_train, 2)

    X_train_final = np.vstack([X_train_scaled, X_aug])
    y_train_final = np.hstack([y_train, y_aug])

    # --------------------------------------------------------
    # RESHAPE
    # --------------------------------------------------------
    X_train_cnn = X_train_final.reshape(
        X_train_final.shape[0],
        X_train_final.shape[1],
        1
    )

    X_test_cnn = X_test_scaled.reshape(
        X_test_scaled.shape[0],
        X_test_scaled.shape[1],
        1
    )

    # --------------------------------------------------------
    # LOAD OR TRAIN MODEL
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print("✔ Loading saved model...")
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"PositionalEncoding": PositionalEncoding}
        )
        scaler = joblib.load(SCALER_PATH)
        label_classes = np.load(LABEL_PATH, allow_pickle=True).tolist()

    else:
        print("⚠ Training Transformer model...")

        model = build_transformer(
            (X_train_cnn.shape[1], 1),
            n_classes
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        model.fit(
            X_train_cnn,
            y_train_final,
            validation_split=0.1,
            epochs=25,
            batch_size=64,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )

        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        np.save(LABEL_PATH, label_classes)

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------
    y_test_pred = np.argmax(
        model.predict(X_test_cnn),
        axis=1
    )

    acc  = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, average="weighted")
    rec  = recall_score(y_test, y_test_pred, average="weighted")
    f1   = f1_score(y_test, y_test_pred, average="weighted")

    print("\n📊 FINAL TEST SET METRICS")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_classes,
        yticklabels=label_classes
    )
    plt.title("Test Set Confusion Matrix – Transformer")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()