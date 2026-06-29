import pandas as pd
import streamlit as st

from phishing_detection.config import METRICS_PATH, RESULTS_DIR
from ui import configure_page, load_json, metric_percent

configure_page("Model Performance", "📈")
st.markdown('<div class="eyebrow">Evaluation</div>', unsafe_allow_html=True)
st.title("Model Performance")
metrics = load_json(METRICS_PATH)

columns = st.columns(5)
for column, (label, key) in zip(columns, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"), ("ROC–AUC", "roc_auc")]):
    column.metric(label, metric_percent(metrics.get(key)))
st.caption(f"Held-out benchmark · {metrics.get('test_samples', '—')} URL/IP pairs · final federated global model")

st.subheader("Accuracy comparison")
comparison = pd.read_csv(RESULTS_DIR / "accuracy_comparison.csv")
st.bar_chart(comparison.set_index("Model"), color="#2ee6a6")

tab1, tab2, tab3 = st.tabs(["Confusion matrix", "ROC curve", "Precision–recall"])
with tab1: st.image(RESULTS_DIR / "confusion_matrix.png", width="stretch")
with tab2: st.image(RESULTS_DIR / "roc_curve.png", width="stretch")
with tab3: st.image(RESULTS_DIR / "precision_recall_curve.png", width="stretch")

st.warning("The bundled benchmark is deterministic and intentionally separable for a reproducible classroom demonstration. These scores are not a production-world efficacy claim; validate against a time-split, independently sourced URL feed before deployment.")
