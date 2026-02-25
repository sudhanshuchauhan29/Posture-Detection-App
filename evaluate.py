# evaluate.py

import json
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

    print("\nModel:", model_name if model_name else model_type)
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall   :", round(rec, 4))
    print("F1-score :", round(f1, 4))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.title("Final Confusion Matrix")
    plt.show()


def main():

    with open("results_store.json", "r") as f:
        results_store = json.load(f)

    results_store = {int(k): v for k, v in results_store.items()}
    le = joblib.load("label_encoder.pkl")

    final_aggregated_metrics(results_store, le, "Hybrid")

if __name__ == "__main__":
    main()