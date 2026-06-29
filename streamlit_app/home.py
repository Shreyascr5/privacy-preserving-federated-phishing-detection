from ui import configure_page, load_json, metric_percent

import streamlit as st

from phishing_detection.config import METADATA_PATH, METRICS_PATH

configure_page("Home")
metrics = load_json(METRICS_PATH)
metadata = load_json(METADATA_PATH)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Federated phishing defense</div>
  <h1>Investigate a website before it becomes an incident.</h1>
  <p>SentinelFL converts a URL and IP address into security signals, applies a privacy-preserving federated classifier, and explains the result in one analyst-ready report.</p>
  <span class="model-badge">● Prediction generated using Federated Global Model</span>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.3, 1], gap="large")
with left:
    st.subheader("From raw indicators to a defensible verdict")
    st.write("No manual feature entry. The engine checks URL structure, credential-themed language, subdomain depth, entropy, transport security, IP scope, and domain/IP consistency.")
    st.page_link("pages/1_Live_URL_Analysis.py", label="Analyze a website", width="stretch")
with right:
    st.markdown("#### System readiness")
    c1, c2 = st.columns(2)
    c1.metric("Global accuracy", metric_percent(metrics.get("accuracy")))
    c2.metric("ROC–AUC", metric_percent(metrics.get("roc_auc")))
    c3, c4 = st.columns(2)
    c3.metric("Federated clients", metadata.get("clients", "—"))
    c4.metric("FL rounds", metadata.get("rounds", "—"))

st.divider()
st.subheader("Detection pipeline")
steps = [
    ("01", "Collect", "URL + IP only"), ("02", "Extract", "19 lexical and network signals"),
    ("03", "Classify", "Flower FedAvg global model"), ("04", "Explain", "Risk score and contributing factors"),
]
columns = st.columns(4)
for column, (number, title, detail) in zip(columns, steps):
    with column:
        st.caption(number)
        st.markdown(f"### {title}")
        st.write(detail)

st.info("Demo tip: try `https://paypal-login-security.xyz/verify/account` with `185.43.22.15`, then compare it with `https://www.python.org/docs` and `8.8.8.8`.")
