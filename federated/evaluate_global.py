"""Evaluate the saved federated global model and write dashboard artifacts."""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.linear_model import LogisticRegression

from federated.train_federated import vectorize
from phishing_detection.config import DATA_DIR, METRICS_PATH, MODEL_PATH, RESULTS_DIR


def main() -> None:
    model = joblib.load(MODEL_PATH)
    x_test, y_test = vectorize(pd.read_csv(DATA_DIR / "test.csv"))
    probability = model.predict_proba(x_test)[:, 1]
    prediction = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction),
        "recall": recall_score(y_test, prediction),
        "f1": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
        "average_precision": average_precision_score(y_test, probability),
        "test_samples": len(y_test),
    }
    train_x, train_y = vectorize(pd.read_csv(DATA_DIR / "train.csv"))
    baseline = LogisticRegression(max_iter=1000, random_state=42).fit(train_x, train_y)
    comparison = pd.DataFrame({
        "Model": ["Centralized baseline", "Federated global model"],
        "Accuracy": [accuracy_score(y_test, baseline.predict(x_test)), metrics["accuracy"]],
    })
    comparison.to_csv(RESULTS_DIR / "accuracy_comparison.csv", index=False)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.txt").write_text(classification_report(y_test, prediction, target_names=["Legitimate", "Phishing"]), encoding="utf-8")

    style = {"figure.facecolor": "#071018", "axes.facecolor": "#0d1822", "text.color": "#dce8ef", "axes.labelcolor": "#b9cad4", "xtick.color": "#8fa5b3", "ytick.color": "#8fa5b3", "axes.edgecolor": "#29404d"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ConfusionMatrixDisplay(confusion_matrix(y_test, prediction), display_labels=["Legitimate", "Phishing"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Federated Global Model")
        fig.tight_layout(); fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180); plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, probability)
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.plot(fpr, tpr, color="#24d4a4", lw=2, label=f"AUC {metrics['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", color="#57717e"); ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "roc_curve.png", dpi=180); plt.close(fig)

        precision, recall, _ = precision_recall_curve(y_test, probability)
        fig, ax = plt.subplots(figsize=(6, 4.5)); ax.plot(recall, precision, color="#4fa3ff", lw=2, label=f"AP {metrics['average_precision']:.3f}"); ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=180); plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
