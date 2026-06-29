from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FederatedLogisticModel:
    """Portable binary classifier produced by Flower FedAvg aggregation."""

    coef_: np.ndarray
    intercept_: np.ndarray
    feature_names: list[str]
    threshold: float = 0.5

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(values).astype(float)
        return matrix @ self.coef_.reshape(-1) + float(self.intercept_[0])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        scores = np.clip(self.decision_function(values), -35, 35)
        phishing = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack((1.0 - phishing, phishing))

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (self.predict_proba(values)[:, 1] >= self.threshold).astype(int)

    def contributions(self, values: np.ndarray) -> np.ndarray:
        return np.atleast_2d(values)[0] * self.coef_.reshape(-1)
