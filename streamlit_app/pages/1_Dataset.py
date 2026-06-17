import streamlit as st
import pandas as pd

st.title("📊 Dataset Overview")

df = pd.read_csv(
    "dataset/processed/phishing_clean.csv"
)

st.write(
    f"Dataset Shape: {df.shape}"
)

st.dataframe(
    df.head()
)

st.subheader(
    "Class Distribution"
)

st.bar_chart(
    df["Result"].value_counts()
)