import pandas as pd
import streamlit as st

from phishing_detection.history import clear_history, read_history
from ui import configure_page

configure_page("Prediction History", "🕘")
st.markdown('<div class="eyebrow">Local audit trail</div>', unsafe_allow_html=True)
st.title("Prediction History")
records = read_history()
if not records:
    st.info("No investigations recorded yet. Results appear here after a live analysis.")
    st.page_link("pages/1_Live_URL_Analysis.py", label="Start an analysis")
else:
    frame = pd.DataFrame(records)
    c1, c2, c3 = st.columns(3)
    c1.metric("Analyses", len(frame))
    c2.metric("High risk", int((frame["risk_level"] == "High Risk").sum()))
    c3.metric("Flagged phishing", int(frame["prediction"].str.contains("Phishing").sum()))
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button("Export history as CSV", frame.to_csv(index=False), "sentinelfl_prediction_history.csv", "text/csv")
    if st.button("Clear local history"):
        clear_history(); st.rerun()
st.caption("History is stored locally in `results/prediction_history.jsonl`; it is never used for federated training.")
