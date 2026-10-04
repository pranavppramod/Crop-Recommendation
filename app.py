import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "crop_recommendation_model.pkl"
METADATA_PATH = BASE_DIR / "config" / "metadata.json"


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


metadata = load_metadata()
model = load_model()


# ============================================================
# EXTRACT METADATA
# ============================================================

project_info = metadata["project"]
dataset_info = metadata["dataset"]
model_info = metadata["model"]
performance = metadata["performance"]
feature_ranges = metadata["feature_ranges"]

feature_names = metadata["deployment"]["input_feature_order"]
classes = dataset_info["classes"]


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main application width */
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Recommendation card */
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(128, 128, 128, 0.08);
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }

    .recommendation-label {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }

    .recommendation-crop {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    /* Section headings */
    .section-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Small information text */
    .info-text {
        color: #6b7280;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #808080;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌱 About the Model")

    st.write(
        "This application uses a tuned Random Forest classifier "
        "to recommend a suitable crop based on soil and "
        "environmental conditions."
    )

    st.divider()

    st.subheader("Model")

    st.write(
        f"**Algorithm:** {model_info['algorithm']}"
    )

    st.write(
        f"**Trees:** {model_info['parameters']['n_estimators']}"
    )

    st.write(
        f"**Split criterion:** {model_info['parameters']['criterion'].title()}"
    )

    st.divider()

    st.subheader("Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "CV Accuracy",
            f"{performance['cv_accuracy'] * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Test Accuracy",
            f"{performance['test_accuracy'] * 100:.2f}%"
        )

    st.metric(
        "Test F1 Score",
        f"{performance['test_f1'] * 100:.2f}%"
    )

    st.divider()

    st.caption(
        f"Dataset: {dataset_info['rows']} samples "
        f"across {dataset_info['num_classes']} crop classes."
    )

    st.caption(
        f"Model version: {project_info['version']}"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Crop Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Enter the soil and environmental conditions below to receive
        a crop recommendation from the trained machine-learning model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">Soil & Environmental Conditions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-text">Enter values using the same units and measurement conventions as the training dataset.</div>',
    unsafe_allow_html=True
)

st.write("")


# ------------------------------------------------------------
# ROW 1 — N, P, K
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=float(feature_ranges["N"]["min"]),
        max_value=float(feature_ranges["N"]["max"]),
        value=50.0,
        step=1.0,
        help="Nitrogen content in the soil."
    )

with col2:
    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=float(feature_ranges["P"]["min"]),
        max_value=float(feature_ranges["P"]["max"]),
        value=50.0,
        step=1.0,
        help="Phosphorus content in the soil."
    )

with col3:
    potassium = st.number_input(
        "Potassium (K)",
        min_value=float(feature_ranges["K"]["min"]),
        max_value=float(feature_ranges["K"]["max"]),
        value=50.0,
        step=1.0,
        help="Potassium content in the soil."
    )


# ------------------------------------------------------------
# ROW 2 — TEMPERATURE, HUMIDITY, pH
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=float(feature_ranges["temperature"]["min"]),
        max_value=float(feature_ranges["temperature"]["max"]),
        value=25.0,
        step=0.1,
        format="%.2f",
        help="Temperature in degrees Celsius."
    )

with col2:
    humidity = st.number_input(
        "Humidity (%)",
        min_value=float(feature_ranges["humidity"]["min"]),
        max_value=float(feature_ranges["humidity"]["max"]),
        value=70.0,
        step=0.1,
        format="%.2f",
        help="Relative humidity percentage."
    )

with col3:
    ph = st.number_input(
        "Soil pH",
        min_value=float(feature_ranges["ph"]["min"]),
        max_value=float(feature_ranges["ph"]["max"]),
        value=6.5,
        step=0.01,
        format="%.2f",
        help="Soil pH value."
    )


# ------------------------------------------------------------
# ROW 3 — RAINFALL
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=float(feature_ranges["rainfall"]["min"]),
        max_value=float(feature_ranges["rainfall"]["max"]),
        value=100.0,
        step=0.1,
        format="%.2f",
        help="Rainfall in millimetres."
    )


# ============================================================
# PREDICTION
# ============================================================

st.write("")

predict_button = st.button(
    "🌱 Recommend Crop",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_data)[0]

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_data)[0]

    predicted_class_index = list(model.classes_).index(prediction)

    confidence = probabilities[predicted_class_index]


    # ========================================================
    # RECOMMENDATION RESULT
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-heading">Recommendation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="recommendation-label">
                Recommended Crop
            </div>
            <div class="recommendation-crop">
                🌱 {prediction.title()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_col1, confidence_col2 = st.columns([1, 2])

    with confidence_col1:
        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

    with confidence_col2:
        st.progress(
            float(confidence),
            text=f"Prediction confidence: {confidence * 100:.2f}%"
        )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("View input values"):

        display_data = pd.DataFrame(
            {
                "Parameter": [
                    "Nitrogen",
                    "Phosphorus",
                    "Potassium",
                    "Temperature",
                    "Humidity",
                    "pH",
                    "Rainfall"
                ],
                "Value": [
                    nitrogen,
                    phosphorus,
                    potassium,
                    temperature,
                    humidity,
                    ph,
                    rainfall
                ],
                "Unit": [
                    "N",
                    "P",
                    "K",
                    "°C",
                    "%",
                    "",
                    "mm"
                ]
            }
        )

        st.dataframe(
            display_data,
            hide_index=True,
            use_container_width=True
        )


    # ========================================================
    # TOP ALTERNATIVE PREDICTIONS
    # ========================================================

    with st.expander("View alternative predictions"):

        probability_df = pd.DataFrame(
            {
                "Crop": model.classes_,
                "Probability": probabilities
            }
        )

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        ).head(5)

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        probability_df = probability_df.reset_index(drop=True)

        probability_df.index += 1

        probability_df.rename(
            columns={
                "Probability": "Probability (%)"
            },
            inplace=True
        )

        st.dataframe(
            probability_df,
            use_container_width=True
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.info(
    """
    **Note:** This application provides machine-learning-based
    crop recommendations based on the patterns learned from the
    training dataset. The recommendation should not be treated as
    a substitute for professional agricultural advice or
    location-specific agronomic assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        Crop Recommendation System · Random Forest Classifier ·
        Model v{project_info['version']}
    </div>
    """,
    unsafe_allow_html=True
)