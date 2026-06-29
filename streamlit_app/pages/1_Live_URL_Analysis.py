from __future__ import annotations

import pandas as pd
import streamlit as st

from phishing_detection.analyzer import analyze_target
from phishing_detection.history import append_history
from ui import configure_page

configure_page("Live URL Analysis", "🔎")
st.markdown('<div class="eyebrow">Live investigation</div>', unsafe_allow_html=True)
st.title("Analyze URL & IP")
st.caption("Enter the two observable indicators. All model features are extracted automatically.")

with st.form("analysis_form"):
    left, right = st.columns([1.8, 1])
    url = left.text_input("Website URL", placeholder="https://example.com/login", help="A scheme is optional; HTTPS is assumed when omitted.")
    ip_address = right.text_input("IP address", placeholder="185.43.22.15")
    network_checks = st.checkbox("Attempt live reverse-DNS and domain/IP resolution", value=False, help="Optional and dependent on network availability. Lexical analysis always runs locally.")
    submitted = st.form_submit_button("Analyze website", type="primary", width="stretch")

if submitted:
    try:
        with st.spinner("Extracting threat signals and querying the federated global model…"):
            report = analyze_target(url, ip_address, network_checks=network_checks)
            append_history(report)
            st.session_state["latest_report"] = report
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))

report = st.session_state.get("latest_report")
if report:
    st.divider()
    risk_class = report.risk_level.split()[0].lower()
    st.markdown(f"### Verdict · <span class='risk-{risk_class}'>{report.risk_level}</span>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk score", f"{report.risk_score}/100")
    m2.metric("Confidence", f"{report.confidence * 100:.1f}%")
    m3.metric("Prediction", "Phishing" if report.phishing_probability >= .5 else "Legitimate")
    m4.metric("Model probability", f"{report.phishing_probability * 100:.1f}%")
    st.markdown(f"<span class='model-badge'>● Prediction generated using {report.model_source}</span>", unsafe_allow_html=True)
    st.markdown(f"## {report.prediction}")

    summary, details = st.columns([1.05, 1], gap="large")
    with summary:
        st.subheader("Risk indicators")
        if report.indicators:
            for indicator in report.indicators:
                st.warning(f"**{indicator['name']}** · {indicator['detail']}")
        else:
            st.success("No material heuristic indicators were detected.")
    with details:
        st.subheader("Website summary")
        f = report.features
        st.dataframe(pd.DataFrame({
            "Check": ["Normalized host", "HTTPS", "URL length", "Subdomains", "Entropy", "Complexity", "IP validity", "IP scope", "Reverse DNS"],
            "Result": [f.host, "Yes" if f.uses_https else "No", f.url_length, f.subdomain_count, f"{f.entropy:.2f}", f"{f.complexity_score:.0f}/100", "Valid" if f.ip_valid else "Invalid", "Public" if f.ip_public else "Private / reserved / invalid", f.reverse_dns or "Not resolved"],
        }), hide_index=True, width="stretch")

    st.subheader("Why the global model decided this")
    factors = pd.DataFrame(report.top_factors)
    factors["contribution"] = factors["impact"].abs()
    st.bar_chart(factors.set_index("factor")["contribution"], horizontal=True, color="#2ee6a6")
    st.dataframe(factors[["factor", "direction", "value", "impact"]], hide_index=True, width="stretch")
    st.caption("Contributions are feature value × global logistic coefficient. The final risk score blends model probability (72%) with transparent security checks (28%).")
