# train.py

import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from config import *
from utils.data_utils import load_dataset, create_splits, encode_labels
from utils.model_utils import get_ml_models, build_cnn, build_hybrid

def main():

    df = load_dataset(DATA_PATH)
    splits = create_splits(df, N_SPLITS, RANDOM_STATE)

    results_store = {}

    for split_id in range(N_SPLITS):

        X_test = splits[split_id][2]
        y_test_raw = splits[split_id][3]

        X_train = np.vstack(
            [splits[j][0] for j in range(N_SPLITS) if j != split_id]
        )

        y_train_raw = np.hstack(
            [splits[j][1] for j in range(N_SPLITS) if j != split_id]
        )

        y_train, le = encode_labels(y_train_raw)
        y_test = le.transform(y_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results_store[split_id+1] = {
            "ML": {},
            "CNN": {},
            "Hybrid": {}
        }

        # ML Training
        ml_models = get_ml_models()
        for name, clf in ml_models.items():
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)

            results_store[split_id+1]["ML"][name] = {
                "y_true": y_test.tolist(),
                "y_pred": preds.tolist()
            }

        # Save encoder only once
        joblib.dump(le, "label_encoder.pkl")

    with open("results_store.json", "w") as f:
        json.dump(results_store, f)

    print("Training completed.")

if __name__ == "__main__":
    main()