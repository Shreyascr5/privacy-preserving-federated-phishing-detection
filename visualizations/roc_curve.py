import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

df = pd.read_csv(
    "dataset/processed/test.csv"
)

X_test = df.drop(
    "Result",
    axis=1
)

y_test = df["Result"]

model = joblib.load(
    "models/baseline_model.pkl"
)

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(6,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/roc_curve.png",
    dpi=300
)

plt.show()

print(f"AUC Score: {roc_auc:.4f}")