from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import HISTORY_PATH


def append_history(report) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": report.features.normalized_url,
        "ip_address": report.features.supplied_ip,
        "prediction": report.prediction,
        "risk_score": report.risk_score,
        "risk_level": report.risk_level,
        "confidence": round(report.confidence * 100, 2),
        "model_source": report.model_source,
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def read_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    records = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))


def clear_history() -> None:
    HISTORY_PATH.unlink(missing_ok=True)
