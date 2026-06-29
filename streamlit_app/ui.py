from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from phishing_detection.config import METADATA_PATH, METRICS_PATH, MODEL_PATH


def configure_page(title: str, icon: str = "🛡️") -> None:
    st.set_page_config(page_title=f"{title} · SentinelFL", page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
      :root { --mint:#2ee6a6; --blue:#53a7ff; --ink:#061017; --panel:#0c1922; }
      [data-testid="stAppViewContainer"] { background: radial-gradient(circle at 85% 5%, #103044 0, #07131b 33%, #050d13 100%); }
      [data-testid="stSidebar"] { background: #07131a; border-right: 1px solid #18303c; }
      .block-container { padding-top: 2.2rem; max-width: 1320px; }
      h1, h2, h3 { letter-spacing: -0.025em; }
      .eyebrow { color:var(--mint); text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
      .hero { padding:2rem 2.2rem; border:1px solid #1a3947; border-radius:22px; background:linear-gradient(135deg,rgba(16,39,51,.96),rgba(7,19,27,.92)); box-shadow:0 18px 50px rgba(0,0,0,.22); margin-bottom:1.5rem; }
      .hero h1 { font-size:3rem; line-height:1.04; margin:.5rem 0 .7rem; max-width:850px; }
      .hero p { color:#a9bdc8; max-width:790px; font-size:1.05rem; }
      .model-badge { display:inline-block; padding:.42rem .72rem; border-radius:999px; background:rgba(46,230,166,.1); border:1px solid rgba(46,230,166,.35); color:#67f2bf; font-size:.78rem; font-weight:700; }
      div[data-testid="stMetric"] { background:rgba(12,25,34,.8); border:1px solid #183542; border-radius:15px; padding:1rem; }
      div[data-testid="stForm"] { background:rgba(10,25,34,.86); border:1px solid #1c3d4d; border-radius:18px; padding:1.2rem; }
      button[kind="primary"] { background:#18b987 !important; border-color:#2ee6a6 !important; color:#04100d !important; font-weight:700 !important; }
      .risk-high { color:#ff6b76; } .risk-medium { color:#ffc857; } .risk-low { color:#43e0a5; }
      .footer-note { color:#6f8794; font-size:.78rem; }
    </style>
    """, unsafe_allow_html=True)
    sidebar_status()


def load_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def sidebar_status() -> None:
    with st.sidebar:
        st.markdown("## ◈ SentinelFL")
        st.caption("Privacy-preserving threat intelligence")
        st.divider()
        if MODEL_PATH.exists():
            st.success("Global model online")
        else:
            st.error("Model artifact missing")
        metadata = load_json(METADATA_PATH)
        if metadata:
            st.caption(f"{metadata.get('clients', 0)} clients · {metadata.get('rounds', 0)} rounds · {metadata.get('strategy', 'FedAvg')}")
        st.divider()
        st.caption("All prediction features are extracted automatically from the submitted URL and IP address.")


def metric_percent(value) -> str:
    return f"{float(value) * 100:.1f}%" if value is not None else "—"
