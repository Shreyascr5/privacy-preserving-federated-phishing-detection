import streamlit as st

st.title(
    "🌐 Federated Analytics"
)

st.markdown(
    """
    ## Communication Reduction

    Standard FL:
    - 20 Updates

    Proposed FL:
    - 7 Updates

    Reduction:
    - 65%
    """
)

st.markdown("---")

st.subheader(
    "Model Comparison"
)

data = {
    "IID FL": 92.88,
    "Non-IID FL": 86.13,
    "DP-FL": 86.14,
    "Proposed": 86.07
}

st.bar_chart(data)