"""SentinelFL Streamlit entry point with explicit six-page navigation."""
import streamlit as st

pages = [
    st.Page("home.py", title="Home", default=True),
    st.Page("pages/1_Live_URL_Analysis.py", title="Live URL Analysis"),
    st.Page("pages/2_Federated_Learning_Dashboard.py", title="Federated Learning Dashboard"),
    st.Page("pages/3_Model_Performance.py", title="Model Performance"),
    st.Page("pages/4_Prediction_History.py", title="Prediction History"),
    st.Page("pages/5_About_Project.py", title="About Project"),
]

st.navigation(pages).run()
