"""Build a URL/IP dataset suitable for direct inference.

With no arguments this creates a deterministic, balanced demonstration benchmark.
Pass --source-csv to normalize a real feed containing URL, IP, and label columns.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from phishing_detection.config import DATA_DIR

BENIGN_DOMAINS = [
    "google.com", "wikipedia.org", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "python.org", "streamlit.io", "cloudflare.com", "mit.edu",
    "bbc.com", "nytimes.com", "mozilla.org", "stackoverflow.com", "linkedin.com",
]
PHISHING_BRANDS = ["paypal", "microsoft", "google", "amazon", "apple", "netflix", "bank", "wallet"]
PHISHING_WORDS = ["secure-login", "account-verify", "signin-update", "auth-support", "confirm-wallet"]
RISK_TLDS = ["xyz", "click", "top", "buzz", "info", "live", "support", "work"]
PATHS = ["home", "docs", "products", "news", "about", "help", "search", "blog"]
PUBLIC_IPS = ["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222", "76.76.2.0"]


def generate_demo_dataset(samples: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for index in range(samples):
        label = index % 2
        # Deliberately non-IID while retaining both classes at every client.
        client_weights = [0.10, 0.20, 0.25, 0.45] if label == 0 else [0.45, 0.25, 0.20, 0.10]
        client_id = rng.choices([1, 2, 3, 4], weights=client_weights, k=1)[0]
        if label == 0:
            domain = rng.choice(BENIGN_DOMAINS)
            scheme = "https" if rng.random() > 0.04 else "http"
            subdomain = rng.choice(["", "www.", "docs.", "support."])
            path = rng.choice(PATHS)
            query = f"?q={rng.choice(['security', 'privacy', 'python', 'research'])}" if rng.random() < 0.2 else ""
            url = f"{scheme}://{subdomain}{domain}/{path}{query}"
            ip = rng.choice(PUBLIC_IPS)
        else:
            brand = rng.choice(PHISHING_BRANDS)
            lure = rng.choice(PHISHING_WORDS)
            tld = rng.choice(RISK_TLDS)
            digits = rng.randint(10, 99999)
            deep = rng.choice(["", "security.", "accounts.security.", "login.verify."])
            scheme = "http" if rng.random() < 0.68 else "https"
            host = f"{deep}{brand}-{lure}-{digits}.{tld}"
            path = rng.choice(["/login", "/verify/account", "/webscr", "/password/recover", "/invoice"])
            redirect = rng.choice(["", "?redirect=https%3A%2F%2Fexample.com", "?continue=signin&auth=1"])
            if rng.random() < 0.08:
                url = f"http://185.43.{rng.randint(1,254)}.{rng.randint(1,254)}{path}"
            elif rng.random() < 0.08:
                url = f"{scheme}://user@{host}{path}{redirect}"
            else:
                url = f"{scheme}://{host}{path}{redirect}"
            ip = f"185.{rng.randint(10,220)}.{rng.randint(1,254)}.{rng.randint(1,254)}"
        rows.append({"url": url, "ip_address": ip, "label": label, "client_id": client_id})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def normalize_source(path: Path, url_column: str, ip_column: str, label_column: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    missing = {url_column, ip_column, label_column} - set(source.columns)
    if missing:
        raise ValueError(f"Missing source columns: {', '.join(sorted(missing))}")
    labels = source[label_column].replace({-1: 1, "phishing": 1, "legitimate": 0, "benign": 0})
    frame = pd.DataFrame({"url": source[url_column], "ip_address": source[ip_column], "label": labels})
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must resolve to 0 (legitimate) and 1 (phishing).")
    frame["client_id"] = (frame.index % 4) + 1
    return frame.dropna().drop_duplicates(subset=["url", "ip_address"]).reset_index(drop=True)


def write_splits(frame: pd.DataFrame, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(frame, test_size=0.2, stratify=frame["label"], random_state=seed)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    frame.to_csv(output_dir / "url_ip_dataset.csv", index=False)
    clients = output_dir / "clients"
    clients.mkdir(exist_ok=True)
    for client_id in range(1, 5):
        client = train[train["client_id"] == client_id]
        client.to_csv(clients / f"client_{client_id}.csv", index=False)
    print(f"Wrote {len(train):,} training and {len(test):,} test records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--ip-column", default="ip_address")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    frame = normalize_source(args.source_csv, args.url_column, args.ip_column, args.label_column) if args.source_csv else generate_demo_dataset(args.samples, args.seed)
    write_splits(frame, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
