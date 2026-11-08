
import streamlit as st
import os
import numpy as np
from datetime import datetime

from utils.snn_model import SNNPredictor
from utils.eeg_processing import extract_recording_features
from utils.monitoring import get_monitoring_status

st.set_page_config(
    page_title="Alzheimer's SNN Monitoring",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Alzheimer's SNN Patient Monitoring")
st.caption(
    "Research decision-support prototype — "
    "not a clinical diagnostic system."
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_predictor():
    return SNNPredictor(model_dir="models")

try:
    predictor = load_predictor()
    st.success("SNN model loaded successfully")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ==========================================
# PATIENT DETAILS
# ==========================================

st.header("Patient Registration")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")

with col2:
    patient_age = st.number_input(
        "Patient Age",
        min_value=1,
        max_value=120,
        value=65
    )

st.divider()

# ==========================================
# EEG ANALYSIS
# ==========================================

st.header("EEG Analysis")

uploaded_file = st.file_uploader(
    "Upload preprocessed EEG windows (.npy)",
    type=["npy"]
)

if uploaded_file is not None:

    try:
        windows = np.load(
            uploaded_file,
            allow_pickle=False
        )

        if (
            windows.ndim != 3 or
            windows.shape[1:] != (19, 1000)
        ):
            st.error(
                "Expected array shape "
                "(number_of_windows, 19, 1000)."
            )

        else:

            st.write("EEG windows:", len(windows))

            if st.button("Analyze EEG"):

                if not patient_name.strip():
                    st.warning("Enter patient name first.")

                else:
                    with st.spinner("Analyzing EEG..."):

                        features = extract_recording_features(
                            windows
                        )

                        result = predictor.predict(
                            features,
                            mc_passes=10
                        )

                    st.subheader("EEG Model Output")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric(
                            "Healthy model score",
                            f"{result['healthy_probability'] * 100:.2f}%"
                        )

                    with c2:
                        st.metric(
                            "Alzheimer's model score",
                            f"{result['alzheimer_probability'] * 100:.2f}%"
                        )

                    st.info(
                        "Research model scores only. "
                        "They are not a medical diagnosis."
                    )

    except Exception as e:
        st.error(f"EEG processing error: {e}")

st.divider()

# ==========================================
# VITAL MONITORING
# ==========================================

st.header("Patient Monitoring")

c1, c2, c3 = st.columns(3)

with c1:
    heart_rate = st.number_input(
        "Heart Rate (BPM)",
        min_value=30,
        max_value=220,
        value=75
    )

with c2:
    gsr_status = st.selectbox(
        "GSR Status",
        ["Normal", "Elevated"]
    )

with c3:
    minutes_since = st.number_input(
        "Minutes Since Last Restroom Visit",
        min_value=0,
        max_value=1440,
        value=60
    )

if st.button("Check Monitoring Status"):

    status = get_monitoring_status(
        heart_rate,
        gsr_status,
        minutes_since
    )

    if status["alert"]:
        st.warning("Monitoring alerts detected")

        for message in status["messages"]:
            st.write("⚠️", message)

    else:
        st.success("No monitoring alerts detected")

st.divider()

st.caption(
    "Alzheimer's SNN Healthcare Monitoring — "
    "Academic Research Prototype"
)
