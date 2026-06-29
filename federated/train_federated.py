"""Train the deployable global model with Flower FedAvg."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss

from phishing_detection.config import DATA_DIR, METADATA_PATH, MODEL_PATH, RESULTS_DIR
from phishing_detection.features import FEATURE_NAMES, extract_features
from phishing_detection.model import FederatedLogisticModel

ROUNDS = 12
LOCAL_EPOCHS = 4
CLIP_NORM = 2.0
NOISE_MULTIPLIER = 0.025
SEED = 42


@dataclass
class ClientState:
    client_id: int
    x: np.ndarray
    y: np.ndarray
    previous_loss: float = float("inf")
    participated: int = 0
    updates_sent: int = 0


def vectorize(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vectors = [extract_features(row.url, row.ip_address).model_vector() for row in frame.itertuples()]
    return np.vstack(vectors), frame["label"].to_numpy(dtype=int)


def local_fit(state: ClientState, global_params: list[np.ndarray], round_number: int):
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0008, learning_rate="constant", eta0=0.08, random_state=SEED + state.client_id)
    model.partial_fit(state.x[:2], state.y[:2], classes=np.array([0, 1]))
    model.coef_ = global_params[0].copy()
    model.intercept_ = global_params[1].copy()
    for _ in range(LOCAL_EPOCHS):
        model.partial_fit(state.x, state.y)

    clean = [model.coef_.copy(), model.intercept_.copy()]
    loss = float(log_loss(state.y, model.predict_proba(state.x), labels=[0, 1]))
    delta = np.concatenate([(clean[0] - global_params[0]).ravel(), (clean[1] - global_params[1]).ravel()])
    norm = float(np.linalg.norm(delta))
    scale = min(1.0, CLIP_NORM / max(norm, 1e-12))
    rng = np.random.default_rng(SEED + round_number * 100 + state.client_id)
    private = []
    for current, original in zip(clean, global_params):
        clipped_delta = (current - original) * scale
        noise = rng.normal(0.0, NOISE_MULTIPLIER * CLIP_NORM / max(len(state.y), 1) ** 0.5, current.shape)
        private.append(original + clipped_delta + noise)

    improvement = state.previous_loss - loss
    send_update = round_number == 1 or improvement > 0.0001 or round_number % 4 == 0
    state.previous_loss = min(state.previous_loss, loss)
    state.participated += 1
    state.updates_sent += int(send_update)
    metrics = {"client_id": state.client_id, "loss": loss, "update_sent": int(send_update), "delta_norm": norm}
    return private, metrics, send_update


def main() -> None:
    client_paths = sorted((DATA_DIR / "clients").glob("client_*.csv"))
    if len(client_paths) < 2:
        raise FileNotFoundError("Client datasets missing. Run: python -m src.build_url_dataset")
    clients = []
    for idx, path in enumerate(client_paths, start=1):
        x, y = vectorize(pd.read_csv(path))
        clients.append(ClientState(idx, x, y))

    global_params = [np.zeros((1, len(FEATURE_NAMES)), dtype=float), np.zeros(1, dtype=float)]
    strategy = FedAvg(
        min_fit_clients=len(clients), min_available_clients=len(clients), fraction_fit=1.0,
        fit_metrics_aggregation_fn=lambda metrics: {
            "updates_sent": sum(int(values.get("update_sent", 0)) for _, values in metrics)
        },
    )
    status = Status(code=Code.OK, message="local client fit complete")
    training_rows = []

    test_x, test_y = vectorize(pd.read_csv(DATA_DIR / "test.csv"))
    for round_number in range(1, ROUNDS + 1):
        results = []
        pending = []
        for state in clients:
            params, metrics, sent = local_fit(state, global_params, round_number)
            pending.append((state, params, metrics))
            if sent:
                fit_res = FitRes(status, ndarrays_to_parameters(params), len(state.y), metrics)
                results.append((None, fit_res))
        if not results:
            state, params, metrics = min(pending, key=lambda item: item[2]["loss"])
            results.append((None, FitRes(status, ndarrays_to_parameters(params), len(state.y), metrics)))
            state.updates_sent += 1
        aggregated, _ = strategy.aggregate_fit(round_number, results, [])
        if aggregated is None:
            raise RuntimeError("Flower FedAvg returned no aggregated parameters")
        global_params = parameters_to_ndarrays(aggregated)
        model = FederatedLogisticModel(global_params[0], global_params[1], FEATURE_NAMES)
        probabilities = model.predict_proba(test_x)[:, 1]
        accuracy = accuracy_score(test_y, probabilities >= 0.5)
        training_rows.append({
            "round": round_number, "accuracy": accuracy,
            "loss": log_loss(test_y, probabilities), "clients_available": len(clients),
            "clients_updated": len(results), "updates_skipped": len(clients) - len(results),
        })
        print(f"Round {round_number:02d}: accuracy={accuracy:.4f}, updates={len(results)}/{len(clients)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).to_csv(RESULTS_DIR / "federated_training.csv", index=False)
    participation = pd.DataFrame([{
        "client": f"Client {state.client_id}", "samples": len(state.y),
        "rounds_participated": state.participated, "updates_sent": state.updates_sent,
        "updates_skipped": state.participated - state.updates_sent,
    } for state in clients])
    participation.to_csv(RESULTS_DIR / "client_participation.csv", index=False)
    total_possible = ROUNDS * len(clients)
    total_sent = int(participation["updates_sent"].sum())
    metadata = {
        "framework": "Flower", "strategy": "FedAvg", "rounds": ROUNDS,
        "clients": len(clients), "local_epochs": LOCAL_EPOCHS,
        "differential_privacy": {
            "mechanism": "L2-clipped client deltas with Gaussian noise",
            "clip_norm": CLIP_NORM, "noise_multiplier": NOISE_MULTIPLIER,
            "accounting": "Demonstration mechanism; no formal epsilon claim",
        },
        "selective_updates": True, "updates_sent": total_sent,
        "updates_possible": total_possible,
        "communication_reduction_percent": round((1 - total_sent / total_possible) * 100, 2),
        "model_artifact": str(MODEL_PATH.relative_to(Path.cwd())),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved federated global model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
