import streamlit as st
import pandas as pd

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Privacy-Preserving Federated Learning",
    page_icon="🔒",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "🔒 Privacy-Preserving Federated Learning for Phishing Website Detection"
)

st.info(
    """
    This project proposes a Privacy-Preserving and Communication-Efficient
    Federated Learning framework for phishing website detection using:

    • Federated Learning (Flower)

    • Non-IID Client Simulation

    • Differential Privacy

    • Selective Updates

    • Communication-Efficient Training
    """
)

st.markdown("---")

# --------------------------------------------------
# PROJECT OVERVIEW
# --------------------------------------------------

st.header("📌 Project Overview")

st.write(
    """
    Traditional phishing detection systems require centralized collection
    of user data, creating privacy risks.

    This project implements a Federated Learning framework that allows
    multiple clients to collaboratively train a phishing detection model
    without sharing raw data.

    Additional privacy is achieved using Differential Privacy, while
    Selective Updates reduce communication overhead.
    """
)

st.markdown("---")

# --------------------------------------------------
# METRICS
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Accuracy",
        value="91.54%"
    )

with col2:
    st.metric(
        label="ROC-AUC",
        value="97.79%"
    )

with col3:
    st.metric(
        label="Communication Reduction",
        value="65%"
    )

st.markdown("---")

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------

st.header("📊 Experimental Results")

try:

    results_df = pd.read_csv(
        "results/accuracy_results.csv"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

except Exception:

    results_df = pd.DataFrame(
        {
            "Method": [
                "IID FL",
                "Non-IID FL",
                "DP-FL",
                "Proposed Method"
            ],
            "Accuracy": [
                92.88,
                86.13,
                86.14,
                86.07
            ]
        }
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

st.markdown("---")

# --------------------------------------------------
# PROJECT CONTRIBUTIONS
# --------------------------------------------------

st.header("🚀 Key Contributions")

st.success(
    """
    ✅ Federated Learning for phishing detection

    ✅ Non-IID client simulation

    ✅ Differential Privacy integration

    ✅ Selective Update mechanism

    ✅ 65% communication reduction

    ✅ ROC-AUC of 97.79%
    """
)

st.markdown("---")

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

st.header("📈 Final Performance Summary")

summary_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Communication Reduction"
        ],
        "Value": [
            "91.54%",
            "87.80%",
            "93.98%",
            "90.78%",
            "97.79%",
            "65%"
        ]
    }
)

st.table(summary_df)

st.markdown("---")

st.caption(
    "M.Tech Data Science Project | Privacy-Preserving Federated Learning for Phishing Website Detection"
)