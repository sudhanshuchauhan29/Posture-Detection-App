# utils/data_utils.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def create_splits(df, n_splits=4, random_state=42):
    X = df.select_dtypes(include=['float64','int64']).iloc[:, :-1].values
    y = df["Label"].values

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    splits = []
    for train_idx, test_idx in skf.split(X, y):
        splits.append((
            X[train_idx], y[train_idx],
            X[test_idx], y[test_idx]
        ))

    return splits

def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le