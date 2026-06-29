import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ====================================
# PAGE TITLE
# ====================================

st.title("🔍 Phishing Website Prediction")

st.write(
    """
    Predict whether a website is
    Legitimate or Phishing using the
    trained Logistic Regression model.
    """
)

# ====================================
# LOAD MODEL
# ====================================

from pathlib import Path
import joblib

ROOT_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(
    ROOT_DIR / "models" / "baseline_model.pkl"
)

scaler = joblib.load(
    ROOT_DIR / "models" / "scaler.pkl"
)


# ====================================
# LOAD DATASET
# ====================================

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

df = pd.read_csv(
    ROOT_DIR / "dataset" / "processed" / "test.csv"
)

# ====================================
# REAL EXAMPLES
# ====================================

legit_sample = (
    df[df["Result"] == 0]
    .drop("Result", axis=1)
    .iloc[0]
)

phishing_sample = (
    df[df["Result"] == 1]
    .drop("Result", axis=1)
    .iloc[0]
)

# ====================================
# REAL EXAMPLES
# ====================================

X = df.drop("Result", axis=1)
y = df["Result"]

X_scaled = scaler.transform(X)

preds = model.predict(X_scaled)

correct_legit = X[
    (y == 0) &
    (preds == 0)
].iloc[0]

correct_phishing = X[
    (y == 1) &
    (preds == 1)
].iloc[0]

legit_sample = correct_legit

phishing_sample = correct_phishing

# ====================================
# DEMO SECTION
# ====================================

st.subheader("Quick Demo")

sample_type = st.selectbox(
    "Choose Example",
    [
        "Legitimate Example",
        "Phishing Example"
    ]
)

if st.button("Run Demo Prediction"):

    if sample_type == "Legitimate Example":
        sample = legit_sample.values.reshape(1, -1)
    else:
        sample = phishing_sample.values.reshape(1, -1)

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(
        sample_scaled
    )[0]

    probability = model.predict_proba(
        sample_scaled
    )[0]

    st.write("### Prediction Result")

    if prediction == 0:
        st.success(
            "✅ Legitimate Website"
        )
    else:
        st.error(
            "⚠️ Phishing Website"
        )

    st.write(
        f"Confidence: {max(probability)*100:.2f}%"
    )

    st.progress(
        float(max(probability))
    )

# ====================================
# MANUAL INPUT SECTION
# ====================================

st.markdown("---")

st.subheader("Manual Prediction")

feature_names = [
    'having_IP_Address',
    'URL_Length',
    'Shortining_Service',
    'having_At_Symbol',
    'double_slash_redirecting',
    'Prefix_Suffix',
    'having_Sub_Domain',
    'SSLfinal_State',
    'Domain_registeration_length',
    'Favicon',
    'port',
    'HTTPS_token',
    'Request_URL',
    'URL_of_Anchor',
    'Links_in_tags',
    'SFH',
    'Submitting_to_email',
    'Abnormal_URL',
    'Redirect',
    'on_mouseover',
    'RightClick',
    'popUpWidnow',
    'Iframe',
    'age_of_domain',
    'DNSRecord',
    'web_traffic',
    'Page_Rank',
    'Google_Index',
    'Links_pointing_to_page',
    'Statistical_report'
]

inputs = []

with st.expander("Enter Feature Values"):

    for feature in feature_names:

        value = st.number_input(
            feature,
            min_value=-1,
            max_value=1,
            value=0,
            step=1
        )

        inputs.append(value)

if st.button("Predict Website"):

    input_array = np.array(
        inputs
    ).reshape(1, -1)

    input_scaled = scaler.transform(
        input_array
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0]

    st.write("### Prediction Result")

    if prediction == 0:

        st.success(
            "✅ Legitimate Website"
        )

    else:

        st.error(
            "⚠️ Phishing Website"
        )

    st.write(
        f"Confidence: {max(probability)*100:.2f}%"
    )

    st.progress(
        float(max(probability))
    )