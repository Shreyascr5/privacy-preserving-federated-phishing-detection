from __future__ import annotations

from dataclasses import dataclass

import joblib

from .config import MODEL_PATH
from .features import FEATURE_NAMES, FeatureResult, extract_features
from .risk import combined_risk, heuristic_indicators


@dataclass
class AnalysisReport:
    features: FeatureResult
    prediction: str
    phishing_probability: float
    confidence: float
    risk_score: int
    risk_level: str
    indicators: list[dict]
    top_factors: list[dict]
    model_source: str = "Federated Global Model"


def load_global_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Federated model is missing. Run: python -m federated.train_federated")
    return joblib.load(MODEL_PATH)


def analyze_target(url: str, ip_address: str, *, network_checks: bool = False) -> AnalysisReport:
    features = extract_features(url, ip_address, network_checks=network_checks)
    model = load_global_model()
    vector = features.model_vector()
    phishing_probability = float(model.predict_proba(vector)[0, 1])
    prediction = "Potential Phishing Website" if phishing_probability >= model.threshold else "Likely Legitimate Website"
    confidence = phishing_probability if phishing_probability >= model.threshold else 1.0 - phishing_probability
    indicators = heuristic_indicators(features)
    risk_score, risk_level = combined_risk(phishing_probability, indicators)
    contributions = model.contributions(vector)
    ranked = sorted(zip(FEATURE_NAMES, contributions, vector), key=lambda item: abs(item[1]), reverse=True)
    top_factors = [
        {"factor": name, "impact": float(impact), "value": float(value), "direction": "raises risk" if impact > 0 else "reduces risk"}
        for name, impact, value in ranked[:6]
    ]
    return AnalysisReport(features, prediction, phishing_probability, confidence, risk_score, risk_level, indicators, top_factors)
