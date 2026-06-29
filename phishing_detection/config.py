from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "dataset" / "url_ip"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
HISTORY_PATH = RESULTS_DIR / "prediction_history.jsonl"
MODEL_PATH = MODEL_DIR / "federated_global_model.joblib"
METADATA_PATH = MODEL_DIR / "federated_metadata.json"
METRICS_PATH = RESULTS_DIR / "model_metrics.json"

for directory in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
