# ============================================================
# REAL-TIME SEQUENCE HYBRID POSTURE DETECTOR
# ============================================================

import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf
from collections import deque

# ============================================================
# PATH HANDLING (IMPORTANT)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "sequence_hybrid_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "sequence_scaler.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "sequence_labels.npy")

# ============================================================
# LOAD MODEL + SCALER + LABELS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ sequence_hybrid_model.h5 not found")

print("✔ Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_classes = np.load(LABELS_PATH, allow_pickle=True)

print("✅ Model loaded successfully")

# ============================================================
# PARAMETERS
# ============================================================

SEQ_LEN = 20
buffer = deque(maxlen=SEQ_LEN)

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

# ============================================================
# EXTRACT EXACT TRAINING FEATURES (10 FEATURES)
# ============================================================

def extract_angles_dataset_matching(L):
    try:
        return np.array([
            calculate_angle(L[23], L[11], L[13]),   # Shoulder
            calculate_angle(L[11], L[13], L[15]),   # Elbow
            calculate_angle(L[11], L[23], L[25]),   # Hip
            calculate_angle(L[23], L[25], L[27]),   # Knee
            calculate_angle(L[25], L[27], L[31]),   # Ankle
            90.0, 90.0, 90.0, 90.0, 90.0            # Ground Angles
        ], dtype=np.float32)
    except:
        return None

# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

print("🎥 Starting webcam... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = result.pose_landmarks.landmark
        pts = [(lm[i].x * w, lm[i].y * h) for i in range(33)]

        angles = extract_angles_dataset_matching(pts)

        if angles is not None:

            scaled = scaler.transform([angles])[0]
            buffer.append(scaled)

            if len(buffer) == SEQ_LEN:

                seq = np.array(buffer).reshape(1, SEQ_LEN, 10)

                pred = model.predict(seq, verbose=0)[0]
                cls = int(np.argmax(pred))
                confidence = float(np.max(pred))

                label = label_classes[cls]

                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Real-Time Sequence Hybrid Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam closed")