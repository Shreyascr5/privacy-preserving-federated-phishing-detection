import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv(
    "dataset/processed/test.csv"
)

X = df.drop(
    "Result",
    axis=1
)

y = df["Result"]

scaler = joblib.load(
    "models/scaler.pkl"
)

X = scaler.transform(X)

model = joblib.load(
    "models/global_model.pkl"
)

preds = model.predict(X)

print(
    "Accuracy:",
    accuracy_score(y, preds)
)

print(
    "Precision:",
    precision_score(y, preds)
)

print(
    "Recall:",
    recall_score(y, preds)
)

print(
    "F1:",
    f1_score(y, preds)
)