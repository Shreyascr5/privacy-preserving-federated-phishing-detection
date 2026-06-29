import pandas as pd
import streamlit as st

from phishing_detection.config import METADATA_PATH, RESULTS_DIR
from ui import configure_page, load_json

configure_page("Federated Learning Dashboard", "🌐")
st.markdown('<div class="eyebrow">Training observability</div>', unsafe_allow_html=True)
st.title("Federated Learning Dashboard")
metadata = load_json(METADATA_PATH)
training = pd.read_csv(RESULTS_DIR / "federated_training.csv")
clients = pd.read_csv(RESULTS_DIR / "client_participation.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aggregation", metadata.get("strategy", "—"))
c2.metric("Clients", metadata.get("clients", "—"))
c3.metric("Rounds", metadata.get("rounds", "—"))
c4.metric("Communication saved", f"{metadata.get('communication_reduction_percent', 0):.1f}%")

st.subheader("Global training progress")
st.line_chart(training.set_index("round")[["accuracy", "loss"]], color=["#2ee6a6", "#53a7ff"])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Client contributions")
    st.bar_chart(clients.set_index("client")[["updates_sent", "updates_skipped"]], color=["#53a7ff", "#344955"])
    st.dataframe(clients, hide_index=True, width="stretch")
with right:
    st.subheader("Communication reduction")
    communication = pd.DataFrame({"Updates": [metadata.get("updates_possible", 0), metadata.get("updates_sent", 0)]}, index=["Standard FedAvg", "Selective FedAvg"])
    st.bar_chart(communication, color="#2ee6a6")
    st.caption("Clients transmit only when local loss improves beyond the selection threshold, with periodic synchronization to avoid starvation.")

st.subheader("Privacy mechanism")
dp = metadata.get("differential_privacy", {})
st.info(f"Each client clips its parameter delta to L2 norm {dp.get('clip_norm', '—')} and adds Gaussian noise (multiplier {dp.get('noise_multiplier', '—')}) before transmission. {dp.get('accounting', '')}")
st.success("The deployable `federated_global_model.joblib` artifact is the final Flower FedAvg aggregation and is the only model used on the Live Analysis page.")
