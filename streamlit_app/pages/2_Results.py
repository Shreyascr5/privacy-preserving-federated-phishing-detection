import streamlit as st
from PIL import Image

st.title("📈 Experimental Results")

st.subheader(
    "Accuracy Comparison"
)

st.image(
    "results/accuracy_comparison.png"
)

st.subheader(
    "Communication Reduction"
)

st.image(
    "results/communication_comparison.png"
)

st.subheader(
    "Confusion Matrix"
)

st.image(
    "results/confusion_matrix.png"
)

st.subheader(
    "ROC Curve"
)

st.image(
    "results/roc_curve.png"
)