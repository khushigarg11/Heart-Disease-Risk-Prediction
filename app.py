import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------------
# Load Saved Files
# ---------------------------------
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("❤️ Health Predictor")
st.sidebar.markdown("### Developed By")
st.sidebar.success("Amam Jain")

# ---------------------------------
# Title
# ---------------------------------
st.markdown(
    """
    <h1 style="text-align:center;color:#E63946;">
        ❤️ Heart Disease Risk Predictor
    </h1>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
Please fill in the information below. The prediction is generated using a trained
Machine Learning model based on the values you provide.
"""
)

st.divider()

# ---------------------------------
# User Inputs
# ---------------------------------
left, right = st.columns(2)

with left:
    age = st.slider("Age (Years)", 18, 100, 40)

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    sex = "M" if gender == "Male" else "F"

    chest_pain_options = {
        "Typical Angina": "TA",
        "Atypical Angina": "ATA",
        "Non-Anginal Pain": "NAP",
        "No Symptoms (Asymptomatic)": "ASY"
    }

    selected_pain = st.selectbox(
        "Chest Pain Category",
        list(chest_pain_options.keys())
    )
    chest_pain = chest_pain_options[selected_pain]

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol Level (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

with right:

    fasting_choice = st.selectbox(
        "Is fasting blood sugar above 120 mg/dL?",
        ["No", "Yes"]
    )
    fasting_bs = 1 if fasting_choice == "Yes" else 0

    resting_ecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    angina_choice = st.selectbox(
        "Do you experience chest pain during exercise?",
        ["No", "Yes"]
    )
    exercise_angina = "Y" if angina_choice == "Yes" else "N"

    oldpeak = st.slider(
        "ST Depression (Oldpeak)",
        0.0,
        6.0,
        1.0,
        step=0.1
    )

    slope_options = {
        "Up Sloping": "Up",
        "Flat": "Flat",
        "Down Sloping": "Down"
    }

    selected_slope = st.selectbox(
        "ST Segment Slope",
        list(slope_options.keys())
    )

    st_slope = slope_options[selected_slope]

st.divider()

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Heart Disease Risk", use_container_width=True):

    user_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([user_data])

    # Add missing columns
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    # Arrange columns exactly as model expects
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
        st.warning(
            "The model indicates a higher likelihood of heart disease. "
            "Please consult a qualified healthcare professional for proper medical evaluation."
        )
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.info(
            "The model indicates a lower likelihood of heart disease. "
            "Maintaining a healthy lifestyle and regular medical check-ups is still recommended."
        )

st.divider()

st.caption("Developed by Amam Jain ❤️")