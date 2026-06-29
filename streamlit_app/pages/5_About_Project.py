import streamlit as st

from ui import configure_page

configure_page("About Project", "ℹ️")
st.markdown('<div class="eyebrow">Project brief</div>', unsafe_allow_html=True)
st.title("About SentinelFL")
st.write("**Privacy-Preserving Federated Learning Based Phishing Website Detection Using URL and IP Analysis**")
st.write("This project demonstrates how multiple data owners can improve a shared phishing classifier without sending raw URL histories to a central server. The final application accepts only a URL and IP address, derives its own signals, and serves the aggregated global model.")

st.subheader("End-to-end architecture")
st.code("""User URL + IP
      ↓
Lexical URL/IP feature extraction (19 normalized signals)
      ↓
Security checks + Flower federated global model
      ↓
Risk score, verdict, confidence, and explanations
      ↓
Streamlit analyst dashboard""", language=None)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("What is real")
    st.markdown("""
- Four independently partitioned, non-IID clients
- Local `SGDClassifier` logistic updates
- Flower `FedAvg` parameter aggregation
- L2 update clipping and Gaussian noise
- Selective transmission and participation logs
- Final inference from the serialized federated global model
""")
with right:
    st.subheader("Responsible limitations")
    st.markdown("""
- The included URL set is a reproducible demonstration benchmark
- Live DNS checks depend on network availability
- URL-only signals cannot inspect page content or certificates
- The DP mechanism has no formal privacy accountant or epsilon claim
- Production use requires a continuously refreshed, independently labeled feed
""")

st.subheader("Recommended research dataset")
st.write("For a thesis-grade evaluation, ingest PhiUSIIL or another maintained URL-level corpus with timestamped labels, preserve a chronological holdout, and enrich it with passively collected IP mappings. The included builder accepts any CSV with URL, IP, and binary label columns, so the prediction and FL contracts do not change.")
st.caption("Built with Python 3.13 · Streamlit · Flower · scikit-learn · pandas · NumPy · joblib")
