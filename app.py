import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

# ---------- Helper: CSV download link ----------
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'

st.title("Heart Disease Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# =====================================================
# TAB 1 – SINGLE PREDICTION
# =====================================================
with tab1:
    st.header("Single Person Prediction")

    age = st.number_input("Age (Years)", min_value=0, max_value=150)
    sex_input = st.selectbox("Sex", ["Male", "Female"])
    chest_pain_input = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
    )
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
    fasting_bs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
    resting_ecg_input = st.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
    st_slope_input = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    # --- Encode exactly as model expects ---
    sex = 1 if sex_input == "Male" else 0
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
    fasting_bs = 1 if fasting_bs_input == "Yes" else 0
    resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
    exercise_angina = 1 if exercise_angina_input == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

    # --- Create DataFrame: **NO Oldpeak column** ---
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "ST_Slope": [st_slope]
    })

    algorithms = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
    modelnames = ["tree.pkl", "LogisticR.pkl", "Randomforest.pkl", "SVM.pkl"]

    if st.button("Submit"):
        st.subheader("Results")
        st.markdown("---")
        for i, modelname in enumerate(modelnames):
            model = pickle.load(open(modelname, "rb"))
            pred = model.predict(input_df.values)[0]
            st.subheader(algorithms[i])
            st.write("Heart Disease Detected" if pred == 1 else "No Heart Disease Detected")
            st.markdown("---")

# =====================================================
# TAB 2 – BULK PREDICTION
# =====================================================
with tab2:
    st.header("Bulk CSV Prediction")

    st.info(
        "1. CSV must have **exactly these 10 columns** (no target column):\n"
        "   'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', "
        "'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope'.\n"
        "2. All values numeric and no missing cells.\n"
        "3. Encodings:\n"
        "   - Sex: 1=Male, 0=Female\n"
        "   - ChestPainType: 0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic\n"
        "   - FastingBS: 0 or 1\n"
        "   - RestingECG: 0=Normal, 1=ST-T, 2=LVH\n"
        "   - ExerciseAngina: 0=No, 1=Yes\n"
        "   - ST_Slope: 0=Upsloping, 1=Flat, 2=Downsloping"
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        expected_columns = [
            "Age","Sex","ChestPainType","RestingBP","Cholesterol",
            "FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"
        ]

        if set(expected_columns).issubset(input_data.columns):
            input_data = input_data[expected_columns].apply(pd.to_numeric, errors="coerce")

            if input_data.isnull().any().any():
                st.warning("Uploaded file has invalid or missing values. Please fix them.")
            else:
                model = pickle.load(open("LogisticR.pkl", "rb"))
                input_data["Prediction-LR"] = model.predict(input_data.values)
                st.subheader("Predictions")
                st.write(input_data)
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
        else:
            st.warning("CSV columns don't match the required 10 features.")
    else:
        st.info("Upload a CSV file to generate predictions.")

# =====================================================
# TAB 3 – MODEL INFORMATION
# =====================================================
with tab3:
    st.header("Model Accuracy Comparison")
    data = {
        "Decision Tree": 80.97,
        "Logistic Regression": 85.86,
        "Random Forest": 84.23,
        "Support Vector Machine": 89.75
    }
    df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])
    fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy")
    st.plotly_chart(fig)
