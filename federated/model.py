# federated/model.py

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = 30


def create_model():
    return LogisticRegression(
        max_iter=1000,
        warm_start=True
    )


def load_client_data(client_id):

    df = pd.read_csv(
        f"dataset/non_iid_clients/client{client_id}.csv"
    )

    X = df.drop(
        "Result",
        axis=1
    )

    y = df["Result"]

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    return X, y