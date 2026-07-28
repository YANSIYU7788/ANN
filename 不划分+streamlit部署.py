from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# =========================
# 1. Application settings
# =========================
st.set_page_config(
    page_title="ANN-based dNCR Prediction",
    page_icon=":brain:",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent

# Recommended GitHub path: saved_models/ANN_IMP1.joblib
# Also supports your current path: saved_models/saved_model/ANN_IMP1.joblib
MODEL_DIR_CANDIDATES = [
    APP_DIR / "saved_model",
    APP_DIR / "saved_models",
    APP_DIR / "saved_models" / "saved_model",
]

FEATURE_COLS = [
    "Operation_Time",
    "MOCA_Score",
    "Age",
    "Nutritional_Risk",
    "Stroke_History",
    "Frailty",
    "Depression",
    "GFR",
]

DISPLAY_FEATURES = [
    "Age",
    "MOCA_Score",
    "Operation_Time",
    "GFR",
    "Nutritional_Risk",
    "Stroke_History",
    "Frailty",
    "Depression",
]

# Optimal ANN threshold calculated from the training results.
ANN_THRESHOLD = 0.398


# =========================
# 2. Load the five ANN models
# =========================
def find_model_dir() -> Path:
    for directory in MODEL_DIR_CANDIDATES:
        expected_files = [directory / f"ANN_IMP{i}.joblib" for i in range(1, 6)]
        if all(path.exists() for path in expected_files):
            return directory

    checked = "\n".join(str(path) for path in MODEL_DIR_CANDIDATES)
    raise FileNotFoundError(
        "ANN model files were not found. Checked these folders:\n" + checked
    )


@st.cache_resource
def load_models():
    model_dir = find_model_dir()
    return [
        joblib.load(model_dir / f"ANN_IMP{i}.joblib")
        for i in range(1, 6)
    ]


def predict_probability(models, data):
    """Calculate the mean probability from five imputation-specific models."""
    frame = pd.DataFrame(data, columns=FEATURE_COLS)
    probabilities = [model.predict_proba(frame)[:, 1] for model in models]
    return np.mean(probabilities, axis=0)


try:
    ann_models = load_models()
except Exception as exc:
    st.error(f"Unable to load ANN models: {exc}")
    st.stop()


# =========================
# 3. Streamlit interface
# =========================
st.title("ANN-based dNCR Prediction")
st.write("Please enter patient characteristics:")

input_data_original = {}

input_data_original["Age"] = st.number_input(
    "Age:", min_value=18.0, max_value=120.0, value=65.0, step=1.0
)
input_data_original["MOCA_Score"] = st.number_input(
    "MOCA Score:", min_value=0.0, max_value=30.0, value=25.0, step=1.0
)
input_data_original["Operation_Time"] = st.number_input(
    "Operation Time (minutes):", min_value=0.0, value=120.0, step=1.0
)
input_data_original["GFR"] = st.number_input(
    "GFR:", min_value=0.0, value=80.0, step=0.1
)

for feature, label in [
    ("Nutritional_Risk", "Nutritional Risk:"),
    ("Stroke_History", "Stroke History:"),
    ("Frailty", "Frailty:"),
    ("Depression", "Depression:"),
]:
    input_data_original[feature] = st.selectbox(
        label,
        options=[0, 1],
        format_func=lambda value: "No" if value == 0 else "Yes",
    )


# =========================
# 4. Prediction (SHAP removed)
# =========================
if st.button("Predict", use_container_width=True):
    input_data = {
        "Operation_Time": input_data_original["Operation_Time"],
        "MOCA_Score": input_data_original["MOCA_Score"],
        "Age": input_data_original["Age"],
        "Nutritional_Risk": input_data_original["Nutritional_Risk"],
        "Stroke_History": input_data_original["Stroke_History"],
        "Frailty": input_data_original["Frailty"],
        "Depression": input_data_original["Depression"],
        "GFR": input_data_original["GFR"],
    }

    X_input = pd.DataFrame([input_data], columns=FEATURE_COLS)

    # StandardScaler is already included in each saved ANN pipeline.
    pred_prob = float(predict_probability(ann_models, X_input)[0])
    pred_label = int(pred_prob >= ANN_THRESHOLD)

    st.divider()
    st.write(f"Predicted probability: {pred_prob:.4f}")
    st.write(f"Decision threshold: {ANN_THRESHOLD:.3f}")
    st.write(f"Predicted result: {'Yes' if pred_label == 1 else 'No'}")

    if pred_label == 1:
        st.error("High risk of dNCR")
    else:
        st.success("Low risk of dNCR")
