# ============================================================
# MATRIX.PY
# Final Aggregated Metrics + Confusion Matrix
# ============================================================

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# LOAD FILES
# ============================================================

def load_results():
    with open("results_store.json", "r") as f:
        results_store = json.load(f)

    # Convert keys to int
    results_store = {int(k): v for k, v in results_store.items()}
    return results_store


def load_encoder():
    return joblib.load("label_encoder.pkl")


# ============================================================
# FINAL AGGREGATED METRICS
# ============================================================

def final_aggregated_metrics(results_store, le, model_type, model_name=None):

    all_y_true = []
    all_y_pred = []

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

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )

    plt.title(f"Final Confusion Matrix – {model_name if model_name else model_type}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    results_store = load_results()
    le = load_encoder()

    # 🔥 Run for all models
    final_aggregated_metrics(results_store, le, "Hybrid")
    final_aggregated_metrics(results_store, le, "CNN")
    final_aggregated_metrics(results_store, le, "ML", "RandomForest")
    final_aggregated_metrics(results_store, le, "ML", "SVM-RBF")
    final_aggregated_metrics(results_store, le, "ML", "LogisticRegression")
    final_aggregated_metrics(results_store, le, "ML", "KNN-5")
    final_aggregated_metrics(results_store, le, "ML", "DecisionTree")


if __name__ == "__main__":
    main()