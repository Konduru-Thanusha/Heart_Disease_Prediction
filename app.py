# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# import plotly.express as px

# # ---------- Helper: CSV download link ----------
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'

# st.title("Heart Disease Predictor")

# # Tabs
# tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# # =====================================================
# # TAB 1 – SINGLE PREDICTION
# # =====================================================
# with tab1:
#     st.header("Single Person Prediction")

#     age = st.number_input("Age (Years)", min_value=0, max_value=150)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox(
#         "Chest Pain Type",
#         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
#     )
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#     cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
#     resting_ecg_input = st.selectbox(
#         "Resting ECG Results",
#         ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
#     )
#     max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
#     exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox(
#         "Slope of Peak Exercise ST Segment",
#         ["Upsloping", "Flat", "Downsloping"]
#     )

#     # --- Encode exactly as model expects ---
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     # --- Create DataFrame: **NO Oldpeak column** ---
#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [sex],
#         "ChestPainType": [chest_pain],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [fasting_bs],
#         "RestingECG": [resting_ecg],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [exercise_angina],
#         "ST_Slope": [st_slope]
#     })

#     algorithms = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
#     modelnames = ["tree.pkl", "LogisticR.pkl", "Randomforest.pkl", "SVM.pkl"]

#     if st.button("Submit"):
#         st.subheader("Results")
#         st.markdown("---")
#         for i, modelname in enumerate(modelnames):
#             model = pickle.load(open(modelname, "rb"))
#             pred = model.predict(input_df.values)[0]
#             st.subheader(algorithms[i])
#             st.write("Heart Disease Detected" if pred == 1 else "No Heart Disease Detected")
#             st.markdown("---")

# # =====================================================
# # TAB 2 – BULK PREDICTION
# # =====================================================
# with tab2:
#     st.header("Bulk CSV Prediction")

#     st.info(
#         "1. CSV must have **exactly these 10 columns** (no target column):\n"
#         "   'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', "
#         "'RestingECG', 'MaxHR', 'ExerciseAngina', 'ST_Slope'.\n"
#         "2. All values numeric and no missing cells.\n"
#         "3. Encodings:\n"
#         "   - Sex: 1=Male, 0=Female\n"
#         "   - ChestPainType: 0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic\n"
#         "   - FastingBS: 0 or 1\n"
#         "   - RestingECG: 0=Normal, 1=ST-T, 2=LVH\n"
#         "   - ExerciseAngina: 0=No, 1=Yes\n"
#         "   - ST_Slope: 0=Upsloping, 1=Flat, 2=Downsloping"
#     )

#     uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#     if uploaded_file is not None:
#         input_data = pd.read_csv(uploaded_file)

#         expected_columns = [
#             "Age","Sex","ChestPainType","RestingBP","Cholesterol",
#             "FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"
#         ]

#         if set(expected_columns).issubset(input_data.columns):
#             input_data = input_data[expected_columns].apply(pd.to_numeric, errors="coerce")

#             if input_data.isnull().any().any():
#                 st.warning("Uploaded file has invalid or missing values. Please fix them.")
#             else:
#                 model = pickle.load(open("LogisticR.pkl", "rb"))
#                 input_data["Prediction-LR"] = model.predict(input_data.values)
#                 st.subheader("Predictions")
#                 st.write(input_data)
#                 st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
#         else:
#             st.warning("CSV columns don't match the required 10 features.")
#     else:
#         st.info("Upload a CSV file to generate predictions.")

# # =====================================================
# # TAB 3 – MODEL INFORMATION
# # =====================================================
# with tab3:
#     st.header("Model Accuracy Comparison")
#     data = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "Support Vector Machine": 89.75
#     }
#     df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])
#     fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy")
#     st.plotly_chart(fig)

















# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# import plotly.express as px

# # =======================
# # CUSTOM STYLING (NEW)
# # =======================
# st.markdown("""
# <style>

# body {
#     background: linear-gradient(135deg, #e2ebf0 0%, #cfd9df 100%);
# }

# .main {
#     background: transparent;
# }

# .block-container {
#     padding-top: 2rem;
# }

# .container-box {
#     background: rgba(255, 255, 255, 0.85);
#     padding: 25px;
#     border-radius: 12px;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.08);
#     margin-bottom: 25px;
# }

# .result-positive {
#     background: #ffebee;
#     padding: 12px;
#     border-left: 6px solid #c62828;
#     border-radius: 6px;
#     font-size: 18px;
#     font-weight: 600;
#     color: #b71c1c;
# }

# .result-negative {
#     background: #e8f5e9;
#     padding: 12px;
#     border-left: 6px solid #2e7d32;
#     border-radius: 6px;
#     font-size: 18px;
#     font-weight: 600;
#     color: #1b5e20;
# }

# </style>
# """, unsafe_allow_html=True)

# # ---------- Helper: CSV download link ----------
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'

# st.title("Heart Disease Prediction Dashboard")


# # Tabs
# tab1, tab2, tab3 = st.tabs(["✔ Predict", "📁 Bulk Predict", "📊 Model Information"])


# # =====================================================
# # TAB 1 – SINGLE PREDICTION
# # =====================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Prediction")

#     age = st.number_input("Age (Years)", min_value=0, max_value=150)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox(
#         "Chest Pain Type",
#         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
#     )
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#     cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
#     resting_ecg_input = st.selectbox(
#         "Resting ECG Results",
#         ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
#     )
#     max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
#     exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox(
#         "Slope of Peak Exercise ST Segment",
#         ["Upsloping", "Flat", "Downsloping"]
#     )

#     # Encodings
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [sex],
#         "ChestPainType": [chest_pain],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [fasting_bs],
#         "RestingECG": [resting_ecg],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [exercise_angina],
#         "ST_Slope": [st_slope]
#     })

#     algorithms = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
#     modelnames = ["tree.pkl", "LogisticR.pkl", "Randomforest.pkl", "SVM.pkl"]

#     if st.button("🔍 Predict"):
#         st.subheader("Results")
#         st.markdown("---")

#         for i, modelname in enumerate(modelnames):
#             model = pickle.load(open(modelname, "rb"))
#             pred = model.predict(input_df.values)[0]

#             st.subheader(algorithms[i])

#             if pred == 1:
#                 st.markdown('<div class="result-positive">❤️ Heart Disease Detected</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="result-negative">💚 No Heart Disease Detected</div>', unsafe_allow_html=True)

#             st.markdown("---")

#     st.markdown('</div>', unsafe_allow_html=True)


# # =====================================================
# # TAB 2 – BULK PREDICTION
# # =====================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Upload CSV for Multiple Predictions")

#     st.info(
#         "CSV must have exactly these 10 columns: Age, Sex, ChestPainType, RestingBP, "
#         "Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, ST_Slope"
#     )

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file is not None:
#         input_data = pd.read_csv(uploaded_file)

#         expected_columns = [
#             "Age","Sex","ChestPainType","RestingBP","Cholesterol",
#             "FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"
#         ]

#         if set(expected_columns).issubset(input_data.columns):
#             input_data = input_data[expected_columns].apply(pd.to_numeric, errors="coerce")

#             if input_data.isnull().any().any():
#                 st.warning("Uploaded file has invalid or missing values.")
#             else:
#                 model = pickle.load(open("LogisticR.pkl", "rb"))
#                 input_data["Prediction-LR"] = model.predict(input_data.values)
#                 st.subheader("Predictions")
#                 st.write(input_data)
#                 st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
#         else:
#             st.warning("CSV columns don't match the required format.")
#     else:
#         st.info("Upload a CSV file to begin.")

#     st.markdown('</div>', unsafe_allow_html=True)


# # =====================================================
# # TAB 3 – MODEL INFORMATION
# # =====================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy Comparison")

#     data = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "Support Vector Machine": 89.75
#     }
#     df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])
#     fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy")
#     st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)









# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# import plotly.express as px

# # ---------- Custom Background Color (CSS) ----------
# page_bg = """
# <style>
#     .stApp {
#         background: linear-gradient(to right, #e8f0ff, #ffffff);
#     }
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # ---------- Helper: Download Predictions CSV ----------
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'


# st.title(" Heart Disease Predictor")

# # Tabs
# tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

# # =====================================================
# # TAB 1 – SINGLE PREDICTION
# # =====================================================
# with tab1:
#     st.header("Single Person Prediction")

#     age = st.number_input("Age (Years)", min_value=0, max_value=150)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox(
#         "Chest Pain Type",
#         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
#     )
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#     cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
#     resting_ecg_input = st.selectbox(
#         "Resting ECG Results",
#         ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
#     )
#     max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
#     exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox(
#         "Slope of Peak Exercise ST Segment",
#         ["Upsloping", "Flat", "Downsloping"]
#     )

#     # --- Encode Inputs ---
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [sex],
#         "ChestPainType": [chest_pain],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [fasting_bs],
#         "RestingECG": [resting_ecg],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [exercise_angina],
#         "ST_Slope": [st_slope]
#     })

#     algorithms = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
#     modelnames = ["tree.pkl", "LogisticR.pkl", "Randomforest.pkl", "SVM.pkl"]

#     if st.button("Submit"):
#         st.subheader("Results")
#         st.markdown("---")
#         for i, modelname in enumerate(modelnames):
#             model = pickle.load(open(modelname, "rb"))
#             pred = model.predict(input_df.values)[0]
#             st.subheader(algorithms[i])
#             st.write(" Heart Disease Detected" if pred == 1 else " No Heart Disease")
#             st.markdown("---")

# # =====================================================
# # TAB 2 – BULK PREDICTION
# # =====================================================
# with tab2:
#     st.header("Bulk CSV Prediction")

#     st.info(
#         "Upload a CSV file with **exactly these 10 columns**:\n"
#         "Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, ST_Slope\n\n"
#         "Encoding:\n"
#         "- Sex: 1=Male, 0=Female\n"
#         "- ChestPainType: 0=Typical, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic\n"
#         "- FastingBS: 0/1\n"
#         "- RestingECG: 0=Normal, 1=ST-T, 2=LVH\n"
#         "- ExerciseAngina: 0=No, 1=Yes\n"
#         "- ST_Slope: 0=Up, 1=Flat, 2=Down"
#     )

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file:
#         input_data = pd.read_csv(uploaded_file)

#         expected_columns = [
#             "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
#             "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "ST_Slope"
#         ]

#         if set(expected_columns).issubset(input_data.columns):
#             input_data = input_data[expected_columns].apply(pd.to_numeric, errors="coerce")

#             if input_data.isnull().any().any():
#                 st.warning("Uploaded file contains invalid or missing values.")
#             else:
#                 model = pickle.load(open("LogisticR.pkl", "rb"))
#                 input_data["Prediction-LR"] = model.predict(input_data.values)
#                 st.subheader("Predictions")
#                 st.write(input_data)
#                 st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
#         else:
#             st.error("CSV columns do NOT match required 10 columns!")

# # =====================================================
# # TAB 3 – MODEL INFORMATION
# # =====================================================
# with tab3:
#     st.header("Model Accuracy Comparison")
    
#     data = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "Support Vector Machine": 89.75
#     }

#     df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])
#     fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy")
#     st.plotly_chart(fig)








# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import base64
# import plotly.express as px

# # ======================================================
# # CUSTOM MODERN UI STYLES (FULL REDESIGN)
# # ======================================================
# st.markdown("""
# <style>

# body {
#     background-color: #87CEEB; /* SKY BLUE */
#     background-size: cover;
#     margin: 0;
#     padding: 0;
#     font-family: 'Segoe UI', sans-serif;
# }

# .main {
#     background: transparent !important;
# }

# h1, h2, h3 {
#     font-weight: 700 !important;
#     color: #003049;
# }

# .container-box {
#     background: rgba(255, 255, 255, 0.85);
#     padding: 30px;
#     border-radius: 16px;
#     box-shadow: 0 8px 25px rgba(0,0,0,0.15);
#     margin-bottom: 30px;
#     backdrop-filter: blur(10px);
# }

# div.stButton > button {
#     background-color: #0288d1;
#     color: white;
#     border-radius: 10px;
#     padding: 12px 16px;
#     font-size: 18px;
#     border: none;
#     transition: 0.2s;
# }

# div.stButton > button:hover {
#     background-color: #01579b;
# }

# .result-positive {
#     background: #ffebee;
#     padding: 14px;
#     border-left: 6px solid #c62828;
#     border-radius: 10px;
#     font-size: 18px;
#     font-weight: 600;
#     color: #b71c1c;
# }

# .result-negative {
#     background: #e8f5e9;
#     padding: 14px;
#     border-left: 6px solid #2e7d32;
#     border-radius: 10px;
#     font-size: 18px;
#     font-weight: 600;
#     color: #1b5e20;
# }

# .upload-box {
#     background: rgba(255, 255, 255, 0.7);
#     padding: 20px;
#     border-radius: 12px;
#     border: 2px dashed #0288d1;
#     text-align: center;
# }

# </style>
# """, unsafe_allow_html=True)

# # ======================================================
# # DOWNLOAD LINK FUNCTION
# # ======================================================
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="font-size:18px;">⬇ Download Predictions CSV</a>'


# # ======================================================
# # TITLE
# # ======================================================
# st.title("💙 Heart Disease Prediction System")
# st.markdown("### *A simple, fast and intelligent tool to check your heart health.*")


# # ======================================================
# # TABS
# # ======================================================
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Info"])


# # ======================================================
# # TAB 1 – SINGLE PREDICTION
# # ======================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Heart Disease Check")

#     age = st.number_input("Age (Years)", min_value=0, max_value=150)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox(
#         "Chest Pain Type",
#         ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
#     )
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#     cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["Yes", "No"])
#     resting_ecg_input = st.selectbox(
#         "Resting ECG Results",
#         ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
#     )
#     max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
#     exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox(
#         "Slope of Peak Exercise ST Segment",
#         ["Upsloping", "Flat", "Downsloping"]
#     )

#     # Encodings
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [sex],
#         "ChestPainType": [chest_pain],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [fasting_bs],
#         "RestingECG": [resting_ecg],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [exercise_angina],
#         "ST_Slope": [st_slope]
#     })

#     algorithms = ["Decision Tree", "Logistic Regression", "Random Forest", "Support Vector Machine"]
#     modelnames = ["tree.pkl", "LogisticR.pkl", "Randomforest.pkl", "SVM.pkl"]

#     if st.button("🔎 Run Prediction"):
#         st.subheader("Prediction Results")
#         st.markdown("---")

#         for i, modelname in enumerate(modelnames):
#             model = pickle.load(open(modelname, "rb"))
#             pred = model.predict(input_df.values)[0]

#             st.subheader(algorithms[i])
#             if pred == 1:
#                 st.markdown('<div class="result-positive">❤️ Heart Disease Detected</div>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<div class="result-negative">💚 No Heart Disease Detected</div>', unsafe_allow_html=True)
#             st.markdown("---")

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 2 – BULK CSV PREDICTION
# # ======================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Bulk Heart Disease Prediction")

#     st.markdown('<div class="upload-box"><h3>📁 Upload CSV File</h3></div>', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("", type=["csv"])

#     if uploaded_file:
#         input_data = pd.read_csv(uploaded_file)

#         expected = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
#                     "FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"]

#         if set(expected).issubset(input_data.columns):
#             input_data = input_data[expected].apply(pd.to_numeric, errors="coerce")

#             if input_data.isnull().any().any():
#                 st.warning("⚠ Some data is missing or invalid. Please correct it.")
#             else:
#                 model = pickle.load(open("LogisticR.pkl", "rb"))
#                 input_data["Prediction-LR"] = model.predict(input_data.values)
#                 st.success("Predictions generated successfully!")
#                 st.write(input_data)
#                 st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
#         else:
#             st.warning("CSV columns do not match required format.")

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 3 – MODEL INFORMATION
# # ======================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy Comparison")

#     data = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }
#     df = pd.DataFrame(list(data.items()), columns=["Model", "Accuracy"])
#     fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy", color="Accuracy")
#     st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)

















# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ======================================================
# # DARK / LIGHT MODE TOGGLE
# # ======================================================
# mode = st.sidebar.radio("Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

# if mode == "🌞 Light Mode":
#     bg_color = "#87CEEB"  # Sky blue
#     text_color = "#003049"
#     card_color = "rgba(255,255,255,0.75)"
# else:
#     bg_color = "#0A192F"
#     text_color = "white"
#     card_color = "rgba(255,255,255,0.10)"

# # ======================================================
# # CUSTOM CSS + ANIMATIONS + UI
# # ======================================================
# st.markdown(f"""
# <style>

# body {{
#     background: {bg_color};
#     background-size: 400% 400%;
#     animation: gradientMove 12s ease infinite;
#     font-family: 'Segoe UI', sans-serif;
#     color: {text_color};
# }}

# @keyframes gradientMove {{
#     0% {{ background-position: 0% 50%; }}
#     50% {{ background-position: 100% 50%; }}
#     100% {{ background-position: 0% 50%; }}
# }}

# h1, h2, h3, h4 {{
#     color: {text_color} !important;
#     font-weight: 700;
# }}

# .container-box {{
#     background: {card_color};
#     padding: 25px;
#     border-radius: 16px;
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
#     margin-bottom: 25px;
#     backdrop-filter: blur(10px);
# }}

# div.stButton > button {{
#     background-color: #0288d1;
#     color: white;
#     border-radius: 12px;
#     font-size: 18px;
#     padding: 10px 18px;
#     border: none;
#     transition: 0.25s;
# }}

# div.stButton > button:hover {{
#     background-color: #01579b;
#     transform: scale(1.05);
# }}

# .heartbeat {{
#     animation: beat 0.8s infinite alternate;
# }}

# @keyframes beat {{
#     from {{ transform: scale(1); }}
#     to   {{ transform: scale(1.25); }}
# }}

# .result-positive {{
#     background: rgba(255,0,0,0.15);
#     padding: 14px;
#     border-left: 6px solid red;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# .result-negative {{
#     background: rgba(0,255,0,0.15);
#     padding: 14px;
#     border-left: 6px solid green;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# </style>
# """, unsafe_allow_html=True)


# # ======================================================
# # DOWNLOAD HELPERS
# # ======================================================
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="font-size:18px;">⬇ Download Predictions CSV</a>'


# # ======================================================
# # PAGE HEADER
# # ======================================================
# st.title("💙 Advanced Heart Disease Prediction System")
# st.markdown("### *Experience a modern, interactive & intelligent way to check your heart health.*")


# # ======================================================
# # TABS
# # ======================================================
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ======================================================
# # TAB 1 - SINGLE PREDICTION
# # ======================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Heart Health Check")

#     # INPUTS
#     age = st.number_input("Age", min_value=1, max_value=120)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#     resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300)
#     cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["Yes", "No"])
#     resting_ecg_input = st.selectbox("Resting ECG", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#     max_hr = st.number_input("Max Heart Rate", 60, 202)
#     exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # ENCODING
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # MODELS
#     model = pickle.load(open("LogisticR.pkl", "rb"))

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df.values)[0]
#         prob = model.predict_proba(input_df.values)[0][1] * 100

#         st.subheader("Prediction Output")

#         # Heartbeat effect
#         st.markdown("<h1 class='heartbeat' style='text-align:center; color:red;'></h1>", unsafe_allow_html=True)

#         if pred == 1:
#             st.markdown("<div class='result-positive'> **Risk of Heart Disease Detected**</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 **No Heart Disease Detected**</div>", unsafe_allow_html=True)

#         # Gauge chart for probability
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=prob,
#             title={'text': "Risk Probability (%)"},
#             gauge={'axis': {'range': [0, 100]},
#                    'bar': {'color': "red" if prob > 50 else "green"}}
#         ))
#         st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 2 — BULK
# # ======================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Upload CSV for Multiple Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#         df["Prediction"] = model.predict(df.values)
#         df["Probability"] = model.predict_proba(df.values)[:, 1] * 100

#         st.success("Predictions Generated")
#         st.write(df)
#         st.markdown(get_binary_file_downloader_html(df), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 3 — MODEL DETAILS
# # ======================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy Chart")

#     accuracy = {"Decision Tree":80.97, "Logistic Regression":85.86, "Random Forest":84.23, "SVM":89.75}
#     df = pd.DataFrame(list(accuracy.items()), columns=["Model","Accuracy"])

#     fig = go.Figure(go.Bar(x=df["Model"], y=df["Accuracy"], marker_color="skyblue"))
#     st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ======================================================
# # DARK / LIGHT MODE TOGGLE
# # ======================================================
# mode = st.sidebar.radio("Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

# if mode == "🌞 Light Mode":
#     bg_color = "rgba(255,255,255,0.60)"  # slight white overlay
#     text_color = "#0A192F"
#     card_color = "rgba(255,255,255,0.75)"
#     image_opacity = "0.35"
# else:
#     bg_color = "rgba(10,25,47,0.70)"      # dark transparent
#     text_color = "white"
#     card_color = "rgba(255,255,255,0.10)"
#     image_opacity = "0.25"


# # ======================================================
# # CUSTOM CSS FOR BACKGROUND IMAGE + UI
# # ======================================================
# st.markdown(f"""
# <style>

# .stApp {{
#     background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
#                 url("heart_bg.jpg");      /* background image */
#     background-size: cover;
#     background-attachment: fixed;
#     background-repeat: no-repeat;
# }}

# body {{
#     color: {text_color};
#     font-family: "Segoe UI";
# }}

# h1, h2, h3, h4 {{
#     color: {text_color} !important;
# }}

# .container-box {{
#     background: {card_color};
#     padding: 25px;
#     border-radius: 16px;
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
#     margin-bottom: 25px;
#     backdrop-filter: blur(8px);
# }}

# div.stButton > button {{
#     background-color: #0288d1;
#     color: white;
#     border-radius: 12px;
#     font-size: 18px;
#     padding: 10px 18px;
#     border: none;
#     transition: 0.25s;
# }}

# div.stButton > button:hover {{
#     background-color: #01579b;
#     transform: scale(1.05);
# }}

# .heartbeat {{
#     animation: beat 0.8s infinite alternate;
# }}

# @keyframes beat {{
#     from {{ transform: scale(1); }}
#     to   {{ transform: scale(1.25); }}
# }}

# .result-positive {{
#     background: rgba(255,0,0,0.25);
#     padding: 14px;
#     border-left: 6px solid red;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# .result-negative {{
#     background: rgba(0,255,0,0.25);
#     padding: 14px;
#     border-left: 6px solid green;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# </style>
# """, unsafe_allow_html=True)



# # ======================================================
# # DOWNLOAD CSV HELPERS
# # ======================================================
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="font-size:18px;">⬇ Download Predictions CSV</a>'


# # ======================================================
# # PAGE HEADER
# # ======================================================
# st.title("💙 Advanced Heart Disease Prediction System")
# st.markdown("### *Modern, intelligent & interactive heart health prediction.*")


# # ======================================================
# # TABS
# # ======================================================
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ======================================================
# # TAB 1 - SINGLE PREDICTION
# # ======================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Prediction")

#     # INPUT FIELDS
#     age = st.number_input("Age", min_value=1, max_value=120)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#     resting_bp = st.number_input("Resting BP (mm Hg)", 0, 300)
#     cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar >120?", ["Yes", "No"])
#     resting_ecg_input = st.selectbox("Resting ECG", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#     max_hr = st.number_input("Max Heart Rate", 60, 202)
#     exercise_angina_input = st.selectbox("Exercise Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # ENCODING
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     model = pickle.load(open("LogisticR.pkl", "rb"))

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         st.subheader("Prediction Result")
#         st.markdown("<h1 class='heartbeat' style='text-align:center; color:red;'>❤️</h1>", unsafe_allow_html=True)

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Probability gauge
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=prob,
#             title={'text': "Probability (%)"},
#             gauge={'axis': {'range': [0, 100]},
#                    'bar': {'color': "red" if prob > 50 else "green"}}
#         ))
#         st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 2 — BULK PREDICTIONS
# # ======================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Bulk CSV Prediction")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#         df["Prediction"] = model.predict(df)
#         df["Probability"] = model.predict_proba(df)[:, 1] * 100

#         st.success("Predictions Generated")
#         st.write(df)
#         st.markdown(get_binary_file_downloader_html(df), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 3 — MODEL INSIGHTS
# # ======================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy")

#     accuracy = {"Decision Tree":80.97, "Logistic Regression":85.86, "Random Forest":84.23, "SVM":89.75}
#     df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])

#     fig = go.Figure(go.Bar(x=df["Model"], y=df["Accuracy"], marker_color="skyblue"))
#     st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)













# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ======================================================
# # DARK / LIGHT MODE TOGGLE
# # ======================================================
# mode = st.sidebar.radio("Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

# if mode == "🌞 Light Mode":
#     bg_color = "rgba(255,255,255,0.60)"  # slight white overlay
#     text_color = "#0A192F"
#     card_color = "rgba(255,255,255,0.75)"
#     image_opacity = "0.35"
# else:
#     bg_color = "rgba(10,25,47,0.70)"      # dark transparent
#     text_color = "white"
#     card_color = "rgba(255,255,255,0.10)"
#     image_opacity = "0.25"


# # ======================================================
# # CUSTOM CSS FOR BACKGROUND IMAGE + UI
# # ======================================================
# st.markdown(f"""
# <style>

# .stApp {{
#     background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
#                 url("heart_bg.jpg");      /* background image */
#     background-size: cover;
#     background-attachment: fixed;
#     background-repeat: no-repeat;
# }}

# body {{
#     color: {text_color};
#     font-family: "Segoe UI";
# }}

# h1, h2, h3, h4 {{
#     color: {text_color} !important;
# }}

# .container-box {{
#     background: {card_color};
#     padding: 25px;
#     border-radius: 16px;
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
#     margin-bottom: 25px;
#     backdrop-filter: blur(8px);
# }}

# div.stButton > button {{
#     background-color: #0288d1;
#     color: white;
#     border-radius: 12px;
#     font-size: 18px;
#     padding: 10px 18px;
#     border: none;
#     transition: 0.25s;
# }}

# div.stButton > button:hover {{
#     background-color: #01579b;
#     transform: scale(1.05);
# }}

# .heartbeat {{
#     animation: beat 0.8s infinite alternate;
# }}

# @keyframes beat {{
#     from {{ transform: scale(1); }}
#     to   {{ transform: scale(1.25); }}
# }}

# .result-positive {{
#     background: rgba(255,0,0,0.25);
#     padding: 14px;
#     border-left: 6px solid red;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# .result-negative {{
#     background: rgba(0,255,0,0.25);
#     padding: 14px;
#     border-left: 6px solid green;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# </style>
# """, unsafe_allow_html=True)



# # ======================================================
# # DOWNLOAD CSV HELPERS
# # ======================================================
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="font-size:18px;">⬇ Download Predictions CSV</a>'


# # ======================================================
# # PAGE HEADER
# # ======================================================
# st.title("💙 Advanced Heart Disease Prediction System")
# st.markdown("### *Modern, intelligent & interactive heart health prediction.*")


# # ======================================================
# # TABS
# # ======================================================
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ======================================================
# # TAB 1 - SINGLE PREDICTION
# # ======================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Prediction")

#     # INPUT FIELDS
#     age = st.number_input("Age", min_value=1, max_value=120)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#     resting_bp = st.number_input("Resting BP (mm Hg)", 0, 300)
#     cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar >120?", ["Yes", "No"])
#     resting_ecg_input = st.selectbox("Resting ECG", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#     max_hr = st.number_input("Max Heart Rate", 60, 202)
#     exercise_angina_input = st.selectbox("Exercise Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # ENCODING
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     model = pickle.load(open("LogisticR.pkl", "rb"))

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         st.subheader("Prediction Result")
#         st.markdown("<h1 class='heartbeat' style='text-align:center; color:red;'>❤️</h1>", unsafe_allow_html=True)

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Probability gauge
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=prob,
#             title={'text': "Probability (%)"},
#             gauge={'axis': {'range': [0, 100]},
#                    'bar': {'color': "red" if prob > 50 else "green"}}
#         ))
#         st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 2 — BULK PREDICTIONS
# # ======================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Bulk CSV Prediction")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#         df["Prediction"] = model.predict(df)
#         df["Probability"] = model.predict_proba(df)[:, 1] * 100

#         st.success("Predictions Generated")
#         st.write(df)
#         st.markdown(get_binary_file_downloader_html(df), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 3 — MODEL INSIGHTS
# # ======================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy")

#     accuracy = {"Decision Tree":80.97, "Logistic Regression":85.86, "Random Forest":84.23, "SVM":89.75}
#     df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])

#     fig = go.Figure(go.Bar(x=df["Model"], y=df["Accuracy"], marker_color="skyblue"))
#     st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)






# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ======================================================
# # DARK / LIGHT MODE TOGGLE
# # ======================================================
# mode = st.sidebar.radio("Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

# if mode == "🌞 Light Mode":
#     bg_color = "rgba(255,255,255,0.60)"  
#     text_color = "#0A192F"
#     card_color = "rgba(255,255,255,0.75)"
#     image_opacity = "0.35"
# else:
#     bg_color = "rgba(10,25,47,0.70)"
#     text_color = "white"
#     card_color = "rgba(255,255,255,0.10)"
#     image_opacity = "0.25"


# # ======================================================
# # CUSTOM CSS
# # ======================================================
# st.markdown(f"""
# <style>

# .stApp {{
#     background: linear-gradient(rgba(0,0,0,0.3), rgba(0,0,0,0.3)),
#                 url("heart_bg.jpg");
#     background-size: cover;
#     background-attachment: fixed;
# }}

# body {{
#     color: {text_color};
#     font-family: "Segoe UI";
# }}

# h1, h2, h3, h4 {{
#     color: {text_color} !important;
# }}

# .container-box {{
#     background: {card_color};
#     padding: 25px;
#     border-radius: 16px;
#     box-shadow: 0px 8px 25px rgba(0,0,0,0.2);
#     margin-bottom: 25px;
#     backdrop-filter: blur(8px);
# }}

# div.stButton > button {{
#     background-color: #0288d1;
#     color: white;
#     border-radius: 12px;
#     font-size: 18px;
#     padding: 10px 18px;
#     border: none;
#     transition: 0.25s;
# }}

# div.stButton > button:hover {{
#     background-color: #01579b;
#     transform: scale(1.05);
# }}

# .heartbeat {{
#     animation: beat 0.8s infinite alternate;
# }}

# @keyframes beat {{
#     from {{ transform: scale(1); }}
#     to   {{ transform: scale(1.25); }}
# }}

# .result-positive {{
#     background: rgba(255,0,0,0.25);
#     padding: 14px;
#     border-left: 6px solid red;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# .result-negative {{
#     background: rgba(0,255,0,0.25);
#     padding: 14px;
#     border-left: 6px solid green;
#     border-radius: 10px;
#     font-size: 18px;
# }}

# </style>
# """, unsafe_allow_html=True)


# # ======================================================
# # DOWNLOAD CSV
# # ======================================================
# def get_binary_file_downloader_html(df):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv" style="font-size:18px;">⬇ Download Predictions CSV</a>'


# # ======================================================
# # PAGE HEADER
# # ======================================================
# st.title("💙 Advanced Heart Disease Prediction System")
# st.markdown("### *Modern, intelligent & interactive heart health prediction.*")


# # ======================================================
# # TABS
# # ======================================================
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ======================================================
# # TAB 1 — SINGLE PREDICTION
# # ======================================================
# with tab1:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Single Person Prediction")

#     age = st.number_input("Age", 1, 120)
#     sex_input = st.selectbox("Sex", ["Male", "Female"])
#     chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
#     resting_bp = st.number_input("Resting BP (mm Hg)", 0, 300)
#     cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600)
#     fasting_bs_input = st.selectbox("Fasting Blood Sugar >120?", ["Yes", "No"])
#     resting_ecg_input = st.selectbox("Resting ECG", ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"])
#     max_hr = st.number_input("Max Heart Rate", 60, 202)
#     exercise_angina_input = st.selectbox("Exercise Angina", ["Yes", "No"])
#     st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     model = pickle.load(open("LogisticR.pkl", "rb"))

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         st.subheader("Prediction Result")
#         st.markdown("<h1 class='heartbeat' style='text-align:center; color:red;'>❤️</h1>", unsafe_allow_html=True)

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=prob,
#             title={'text': "Probability (%)"},
#             gauge={'axis': {'range': [0, 100]},
#                    'bar': {'color': "red" if prob > 50 else "green"}}
#         ))
#         st.plotly_chart(fig)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 2 — BULK CSV PREDICTION
# # ======================================================
# with tab2:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Bulk CSV Prediction")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#         df["Prediction"] = model.predict(df)
#         df["Probability"] = model.predict_proba(df)[:, 1] * 100

#         st.success("Predictions Generated")
#         st.write(df)
#         st.markdown(get_binary_file_downloader_html(df), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ======================================================
# # TAB 3 — MODEL INSIGHTS (UPDATED COLORS)
# # ======================================================
# with tab3:
#     st.markdown('<div class="container-box">', unsafe_allow_html=True)
#     st.header("Model Accuracy")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])

#     # COLOR FUNCTION
#     def get_color(acc):
#         if acc >= 85:
#             return "green"
#         elif acc >= 75:
#             return "yellow"
#         else:
#             return "red"

#     colors = [get_color(a) for a in df["Accuracy"]]

#     fig = go.Figure(go.Bar(
#         x=df["Model"],
#         y=df["Accuracy"],
#         marker_color=colors,
#         text=df["Accuracy"],
#         textposition="outside"
#     ))

#     fig.update_layout(
#         yaxis_title="Accuracy (%)",
#         title="Model Performance Comparison",
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         font=dict(color=text_color)
#     )

#     st.plotly_chart(fig)
#     st.markdown('</div>', unsafe_allow_html=True)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"           # very light sky tint (top)
# BG_BOTTOM = "#e9f7ff"        # very light sky tint (bottom)
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"           # iOS blue accent
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"             # dark-blue text
# SUBTEXT = "#54657A"          # muted subtext
# SHADOW = "rgba(10,132,255,0.08)"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* subtle section title */
# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# /* iOS style primary button */
# div.stButton > button, button[kind="primary"] {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* secondary flat button */
# a.stDownloadButton > button {{
#     background: transparent;
#     color: {ACCENT};
#     border-radius: 12px;
#     border: 1px solid {ACCENT};
#     padding: 8px 14px;
#     font-weight: 600;
# }}

# /* input boxes styling (works mostly in modern Streamlit CSS classes) */
# .css-1d391kg input, .css-1d391kg textarea, .css-1d391kg select {{
#     border-radius: 10px;
#     padding: 10px;
# }}

# /* small result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# /* subtle heartbeat emoji styling */
# .heartbeat {{
#     font-size: 42px;
#     line-height: 1;
#     display:block;
#     margin: 6px auto 0;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}

# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")
#     st.write("")  # spacing

#     # Inputs grouped in two-column layout
#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", min_value=1, max_value=120, value=45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, value=120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, value=200)
#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, value=150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode inputs
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model (make sure LogisticR.pkl is in same folder)
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except Exception as e:
#         st.error(f"Model load error: {e}. Make sure 'LogisticR.pkl' exists in the app folder.")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model is None:
#             st.warning("Model not loaded — can't predict.")
#         else:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#             # Gauge: soft iOS colored
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                     'threshold': {
#                         'line': {'color': "red", 'width': 3},
#                         'thickness': 0.75,
#                         'value': 50
#                     }
#                 },
#             ))
#             fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 2 — Bulk predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")
#     st.markdown('<div class="section-subtitle">CSV should contain columns in the same order as the model expects (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, ST_Slope).</div>', unsafe_allow_html=True)

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if model is None:
#                 st.error("Model not loaded — cannot generate predictions.")
#             else:
#                 # We assume the CSV columns match model input order
#                 preds = model.predict(df.values)
#                 probs = model.predict_proba(df.values)[:, 1] * 100
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs, 2)
#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Failed to read or predict: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 3 — Model insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {"Decision Tree":80.97, "Logistic Regression":85.86, "Random Forest":84.23, "SVM":89.75}
#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.write("Summary of validation accuracies (example values):")
#     st.dataframe(acc_df, use_container_width=True)

#     # Bar chart in soft iOS color
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=ACCENT, line=dict(color='rgba(255,255,255,0.6)', width=0.6)),
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))
#     fig.update_layout(yaxis=dict(range=[0,100]), margin=dict(t=10, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer / small print
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis. Consult a qualified professional for health decisions.
# </div>
# """, unsafe_allow_html=True)
















# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"                # iOS sky blue
# DEEP_BLUE = "#0047AB"             # NEW: deep blue for model insights bar chart
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# .heartbeat {{
#     font-size: 42px;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)


# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'


# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode inputs
#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [1 if sex_input == "Male" else 0],
#         "ChestPainType": [["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [1 if fasting_bs_input == "Yes" else 0],
#         "RestingECG": [["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [1 if exercise_angina_input == "Yes" else 0],
#         "ST_Slope": [["Upsloping","Flat","Downsloping"].index(st_slope_input)]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except:
#         model = None
#         st.error("Model load error. Ensure LogisticR.pkl is in the app folder.")

#     if st.button("💙 Predict Now"):
#         if model:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#             # Gauge
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                     'threshold': {
#                         'line': {'color': "red", 'width': 3}, 'thickness': 0.75,
#                         'value': 50
#                     }
#                 }
#             ))
#             fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 2 — Bulk Predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

#     if uploaded_file and model:
#         df = pd.read_csv(uploaded_file)
#         preds = model.predict(df.values)
#         probs = model.predict_proba(df.values)[:, 1] * 100

#         df_out = df.copy()
#         df_out["Prediction"] = preds
#         df_out["Probability (%)"] = np.round(probs, 2)

#         st.success("Predictions completed")
#         st.dataframe(df_out, use_container_width=True)
#         st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 3 — Model Insights (UPDATED COLOR)
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }
    
#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])

#     st.dataframe(acc_df, use_container_width=True)

#     # *** UPDATED: Deep Blue Color ***
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=DEEP_BLUE),     # NEW COLOR
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))

#     fig.update_layout(
#         yaxis=dict(range=[0, 100]),
#         margin=dict(t=10, b=20, l=20, r=20),
#         paper_bgcolor="rgba(0,0,0,0)"
#     )

#     st.plotly_chart(fig, use_container_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)









# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"           # very light sky tint (top)
# BG_BOTTOM = "#e9f7ff"        # very light sky tint (bottom)
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"           # iOS blue accent
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"             # dark-blue text
# SUBTEXT = "#54657A"          # muted subtext
# SHADOW = "rgba(10,132,255,0.08)"

# # 🔵 Deep Blue for Model Insights
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* subtle section title */
# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# /* iOS style primary button */
# div.stButton > button, button[kind="primary"] {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# /* heartbeat icon */
# .heartbeat {{
#     font-size: 42px;
#     line-height: 1;
#     display:block;
#     margin: 6px auto 0;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except Exception as e:
#         st.error(f"Model load error: {e}")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model is None:
#             st.warning("Model not loaded.")
#         else:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", 
#                         unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)

#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                 },
#             ))
#             fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
#                               paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if model:
#                 preds = model.predict(df.values)
#                 probs = model.predict_proba(df.values)[:, 1] * 100
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs, 2)

#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {
#         "Decision Tree":80.97, 
#         "Logistic Regression":85.86, 
#         "Random Forest":84.23, 
#         "SVM":89.75
#     }

#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # 🔵 Deep blue bar chart
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE),
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))

#     fig.update_layout(
#         yaxis=dict(range=[0,100]),
#         margin=dict(t=10, b=20, l=20, r=20),
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)







# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier

# # -------------------------------------------
# # iOS Theme Styling
# # -------------------------------------------
# IOS_BG = "#F2F2F7"
# IOS_CARD = "rgba(255,255,255,0.72)"
# IOS_SHADOW = "0px 4px 20px rgba(0,0,0,0.08)"
# DEEP_BLUE = "#003E92"

# st.set_page_config(page_title="Health Risk Predictor - iOS Theme", layout="wide")

# st.markdown(
#     f"""
#     <style>
#         body {{ background-color: {IOS_BG}; }}
#         .ios-card {{
#             background: {IOS_CARD};
#             padding: 25px;
#             border-radius: 20px;
#             box-shadow: {IOS_SHADOW};
#             backdrop-filter: blur(10px);
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown("<h1 style='text-align:center;'>Health Risk Prediction – iOS Edition</h1>", unsafe_allow_html=True)

# # Tabs
# input_tab, meter_tab, insights_tab = st.tabs(["User Input", "Risk Meter", "Model Insights"])

# # -------------------------------------------
# # USER INPUT TAB
# # -------------------------------------------
# with input_tab:
#     st.markdown("### Enter Your Health Data")

#     age = st.slider("Age", 18, 90, 30)
#     cholesterol = st.slider("Cholesterol Level", 100, 350, 180)
#     bp = st.slider("Blood Pressure", 80, 200, 120)

#     features = pd.DataFrame({
#         "age": [age],
#         "cholesterol": [cholesterol],
#         "bp": [bp]
#     })

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(features)

#     # Dummy models for demo
#     logistic = LogisticRegression()
#     decision = DecisionTreeClassifier()
#     randomf = RandomForestClassifier()
#     svm = SVC(probability=True)

#     # Fake training (placeholder)
#     dummy_X = [[30, 170, 120], [50, 220, 150], [40, 200, 130]]
#     dummy_y = [0, 1, 1]

#     logistic.fit(dummy_X, dummy_y)
#     decision.fit(dummy_X, dummy_y)
#     randomf.fit(dummy_X, dummy_y)
#     svm.fit(dummy_X, dummy_y)

#     # Prediction probability from RandomForest
#     prediction_proba = randomf.predict_proba(X_scaled)[0][1]
#     st.session_state["risk_value"] = prediction_proba

# # -------------------------------------------
# # RISK METER TAB
# # -------------------------------------------
# with meter_tab:
#     st.markdown("### Risk Probability Meter")

#     risk_probability = st.session_state.get("risk_value", 0.5)

#     if risk_probability > 0.70:
#         meter_color = "red"
#     elif risk_probability > 0.40:
#         meter_color = "yellow"
#     else:
#         meter_color = "blue"

#     risk_value = risk_probability * 100

#     gauge = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=risk_value,
#             number={'suffix': "%"},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': meter_color},
#                 'steps': [
#                     {'range': [0, 40], 'color': "rgba(0,102,255,0.25)"},
#                     {'range': [40, 70], 'color': "rgba(255,204,0,0.25)"},
#                     {'range': [70, 100], 'color': "rgba(255,0,0,0.25)"}
#                 ]
#             }
#         )
#     )
#     st.plotly_chart(gauge, use_container_width=True)

# # -------------------------------------------
# # MODEL INSIGHTS TAB
# # -------------------------------------------
# with insights_tab:
#     st.markdown("### Model Performance Insights")

#     acc_df = pd.DataFrame({
#         "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
#         "Accuracy": [85, 78, 90, 88]
#     })

#     fig = go.Figure(
#         go.Bar(
#             x=acc_df["Model"],
#             y=acc_df["Accuracy"],
#             marker=dict(
#                 color=DEEP_BLUE,
#                 line=dict(color="rgba(255,255,255,0.8)", width=1)
#             ),
#             hovertemplate='%{y:.2f}%<extra></extra>'
#         )
#     )

#     fig.update_layout(
#         xaxis_title="Model",
#         yaxis_title="Accuracy (%)",
#         margin=dict(l=20, r=20, t=40, b=20),
#         height=350
#     )

#     st.plotly_chart(fig, use_container_width=True)









# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"

# # deep blue for chart
# INSIGHT_BLUE = "#003A92"

# # ---------------------------
# # Custom CSS
# # ---------------------------
# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
#   color: {TEXT};
# }}

# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: 0.15s ease;
# }}

# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# .result-positive {{
#     background: linear-gradient(90deg, #fff2f2, #ffeaea);
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}

# .result-negative {{
#     background: linear-gradient(90deg, #f0fff5, #e8fff0);
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# .heartbeat {{
#     font-size: 42px;
#     margin: 6px auto 0;
#     display:block;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # CSV Download Helper
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download Predictions CSV</a>'

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:rgba(10,132,255,0.12); width:54px; height:54px; 
#               border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky UI • Clean & Minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ----------------------------------------------------
# # TAB 1 — Single Prediction
# # ----------------------------------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns(2)
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#                ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting BP (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#                ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except:
#         st.error("❌ Model file 'LogisticR.pkl' not found!")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model:
#             pred = int(model.predict(input_df)[0])
#             prob = float(model.predict_proba(input_df)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ High Risk of Heart Disease</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=prob,
#                 gauge={'axis': {'range': [0, 100]}, 'bar': {'color': ACCENT}},
#                 title={'text': "Risk Probability (%)"}
#             ))
#             fig.update_layout(margin=dict(t=0, b=0))
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ----------------------------------------------------
# # TAB 2 — Bulk Prediction
# # ----------------------------------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Prediction")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         df = pd.read_csv(uploaded_file)
#         try:
#             preds = model.predict(df.values)
#             probs = model.predict_proba(df.values)[:, 1] * 100
#             df_out = df.copy()
#             df_out["Prediction"] = preds
#             df_out["Probability"] = np.round(probs, 2)

#             st.success("✔ Predictions Complete")
#             st.dataframe(df_out, use_container_width=True)
#             st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"CSV Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ----------------------------------------------------
# # TAB 3 — Model Insights (FIXED ERROR)
# # ----------------------------------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy Comparison")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE)
#     ))

#     fig.update_layout(yaxis=dict(range=[0, 100]), margin=dict(t=10, b=20))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not medical advice.
# </div>
# """, unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import numpy as np

# # -------------------------------
# # PAGE CONFIG & iOS THEME
# # -------------------------------
# st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# st.markdown("""
# <style>
# /* iOS Glass Card */
# .card {
#     background: rgba(255,255,255,0.55);
#     backdrop-filter: blur(12px);
#     border-radius: 22px;
#     padding: 22px;
#     box-shadow: 0px 4px 18px rgba(0,0,0,0.12);
#     margin-bottom: 20px;
# }
# .big-title {
#     font-size: 34px;
#     font-weight: 700;
# }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # HEADER
# # -------------------------------
# st.markdown('<p class="big-title">💓 Heart Disease Risk Analyzer (iOS Theme)</p>', unsafe_allow_html=True)
# st.write("Predict heart disease risk using clinical inputs and ML models.")

# tab1, tab2, tab3 = st.tabs(["🏥 User Input", "📊 Prediction", "📘 Model Insights"])

# # ============================================================
# # TAB 1 – USER INPUT SECTION
# # ============================================================
# with tab1:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Patient Input Data")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", min_value=1, max_value=120)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_binary = st.selectbox("Chest Pain (Yes/No)", ["Yes", "No"])
#         chest_pain_input = st.selectbox(
#             "Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
#         )
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar (>120 mg/dl)", ["Yes", "No"])
#         resting_ecg_binary = st.selectbox("Resting ECG (Yes/No)", ["Yes", "No"])

#     with col2:
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
#         cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600)
#         resting_ecg_input = st.selectbox(
#             "Resting ECG Results",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"]
#         )
#         max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=202)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
#         st_slope_input = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

#     # -------------------------------
#     # ENCODE INPUTS
#     # -------------------------------
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     chest_pain_binary_val = 1 if chest_pain_binary == "Yes" else 0
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     resting_ecg_binary_val = 1 if resting_ecg_binary == "Yes" else 0
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     # Create DataFrame for model
#     input_df = pd.DataFrame({
#         "Age": [age],
#         "Sex": [sex],
#         "ChestPainType": [chest_pain],
#         "ChestPainBinary": [chest_pain_binary_val],
#         "RestingBP": [resting_bp],
#         "Cholesterol": [cholesterol],
#         "FastingBS": [fasting_bs],
#         "RestingECG": [resting_ecg],
#         "RestingECG_Binary": [resting_ecg_binary_val],
#         "MaxHR": [max_hr],
#         "ExerciseAngina": [exercise_angina],
#         "ST_Slope": [st_slope]
#     })

#     st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================
# # TAB 2 – PREDICTION
# # ============================================================
# with tab2:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Prediction Result")

#     # Dummy model probability (Replace with actual model)
#     risk_probability = np.random.uniform(0.1, 0.95)

#     # Dynamic Color Logic
#     if risk_probability < 0.33:
#         meter_color = "blue"
#         risk_label = "Low Risk"
#     elif risk_probability < 0.66:
#         meter_color = "yellow"
#         risk_label = "Moderate Risk"
#     else:
#         meter_color = "red"
#         risk_label = "High Risk"

#     # Risk Meter Gauge
#     fig = go.Figure(
#         go.Indicator(
#             mode="gauge+number",
#             value=risk_probability * 100,
#             gauge={
#                 "axis": {"range": [0, 100]},
#                 "bar": {"color": meter_color},
#                 "steps": [
#                     {"range": [0, 33], "color": "rgba(0,0,255,0.3)"},
#                     {"range": [33, 66], "color": "rgba(255,255,0,0.3)"},
#                     {"range": [66, 100], "color": "rgba(255,0,0,0.3)"}
#                 ],
#             }
#         )
#     )

#     st.plotly_chart(fig, use_container_width=True)
#     st.success(f"Risk Level: **{risk_label}** ({risk_probability*100:.1f}%)")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ============================================================
# # TAB 3 – MODEL INSIGHTS
# # ============================================================
# with tab3:
#     st.markdown('<div class="card">', unsafe_allow_html=True)
#     st.subheader("Model Performance Insights")

#     acc_df = pd.DataFrame({
#         "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
#         "Accuracy": [85, 78, 90, 88]
#     })

#     fig = go.Figure(
#         go.Bar(
#             x=acc_df["Model"],
#             y=acc_df["Accuracy"],
#             marker=dict(
#                 color="#003E92",
#                 line=dict(color="white", width=1)
#             ),
#             hovertemplate='%{y:.2f}%<extra></extra>'
#         )
#     )
#     fig.update_layout(
#         xaxis_title="Model",
#         yaxis_title="Accuracy (%)",
#         height=350,
#         margin=dict(l=10, r=10, t=30, b=10)
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"           # very light sky tint (top)
# BG_BOTTOM = "#e9f7ff"        # very light sky tint (bottom)
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"           # iOS blue accent
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"             # dark-blue text
# SUBTEXT = "#54657A"          # muted subtext
# SHADOW = "rgba(10,132,255,0.08)"

# # 🔵 Deep Blue for Model Insights
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* subtle section title */
# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# /* iOS style primary button */
# div.stButton > button, button[kind="primary"] {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# /* heartbeat icon */
# .heartbeat {{
#     font-size: 42px;
#     line-height: 1;
#     display:block;
#     margin: 6px auto 0;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except Exception as e:
#         st.error(f"Model load error: {e}")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model is None:
#             st.warning("Model not loaded.")
#         else:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", 
#                         unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)

#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                 },
#             ))
#             fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
#                               paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if model:
#                 preds = model.predict(df.values)
#                 probs = model.predict_proba(df.values)[:, 1] * 100
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs, 2)

#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {
#         "Decision Tree":80.97, 
#         "Logistic Regression":85.86, 
#         "Random Forest":84.23, 
#         "SVM":89.75
#     }

#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # 🔵 Deep blue bar chart
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE),
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))

#     fig.update_layout(
#         yaxis=dict(range=[0,100]),
#         margin=dict(t=10, b=20, l=20, r=20),
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import plotly.express as px
# import base64
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import accuracy_score, confusion_matrix

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI (Enhanced)",
#     layout="centered",
#     initial_sidebar_state="expanded",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient + subtle animated gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
#   background-size: 200% 200%;
#   animation: bgMove 18s ease infinite;
# }}

# @keyframes bgMove {{
#   0% {{ background-position: 0% 50%; }}
#   50% {{ background-position: 100% 50%; }}
#   100% {{ background-position: 0% 50%; }}
# }}

# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* floating hearts */
# .floating-heart {{
#   position: fixed;
#   left: 10px;
#   bottom: 40px;
#   opacity: 0.8;
#   transform: translateY(0);
#   animation: floatUp 6s linear infinite;
# }}
# @keyframes floatUp {{
#   0% {{ transform: translateY(0) rotate(-10deg); opacity:0.9; }}
#   50% {{ transform: translateY(-18px) rotate(10deg); opacity:0.6; }}
#   100% {{ transform: translateY(0) rotate(-10deg); opacity:0.9; }}
# }}

# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
# }}

# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# .patient-summary {{
#     background: rgba(255,255,255,0.95);
#     border-radius: 12px;
#     padding: 12px;
#     border: 1px solid rgba(10,132,255,0.08);
# }}

# .small-muted {{
#     color: {SUBTEXT};
#     font-size:13px;
# }}
# </style>
# """, unsafe_allow_html=True)

# # Floating heart (decorative)
# st.markdown('<img class="floating-heart" src="https://i.imgur.com/0X4Qk2h.png" width="80" alt="heart">', unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <img src="https://i.imgur.com/6kQ2QWc.png" width="28"> 
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal — now feature-rich</div>
#   </div>
#   <div style="margin-left:auto; text-align:right;">
#     <div class="small-muted">Not a medical device</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights", "✨ Extras"])

# # ---------------------------
# # Utility: Load dataset (for charts & fallback training)
# # ---------------------------
# @st.cache_data
# def load_dataset():
#     # try local file first
#     try:
#         df = pd.read_csv("heart.csv")
#     except Exception:
#         # fallback to remote (works when user has internet)
#         df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")
#     return df

# data_df = load_dataset()

# # ---------------------------
# # Utility: Try load model, else train fallback model
# # ---------------------------
# @st.cache_resource
# def get_model_and_scaler():
#     model = None
#     scaler = StandardScaler()
#     # Try to load user's LogisticR.pkl (as original code did)
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#         # If loaded model expects named features, we won't scale here (assume model already prepared).
#         return model, None, "loaded"
#     except Exception:
#         # Train fallback RandomForest on available dataset
#         df = data_df.copy()
#         if "target" in df.columns:
#             X = df.drop(columns=["target"])
#             y = df["target"]
#         else:
#             # if dataset has different schema, try known columns
#             X = df.iloc[:, :-1]
#             y = df.iloc[:, -1]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
#         scaler.fit(X_train)
#         X_train_s = scaler.transform(X_train)
#         rf = RandomForestClassifier(n_estimators=200, random_state=42)
#         rf.fit(X_train_s, y_train)
#         return rf, scaler, "trained"

# model, scaler, model_source = get_model_and_scaler()

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     # Input layout
#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)
#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Patient summary card (UI feature #4)
#     st.markdown('<div class="patient-summary">', unsafe_allow_html=True)
#     st.markdown("<strong>Patient Summary</strong>")
#     st.write(f"Age: {age}  •  Sex: {sex_input}  •  Chest Pain: {chest_pain_input}")
#     st.write(f"Resting BP: {resting_bp} mmHg  •  Cholesterol: {cholesterol} mg/dl")
#     st.write(f"Max HR: {max_hr}  •  Exercise Angina: {exercise_angina_input}  •  Fasting BS: {fasting_bs_input}")
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Encode (keep same order as original)
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Predict button
#     if st.button("💙 Predict Now"):
#         # Prepare features for model (handle scaler if trained fallback)
#         try:
#             if model_source == "trained" and scaler is not None:
#                 X_input = scaler.transform(input_df)
#             else:
#                 X_input = input_df.values  # assume model handles feature names/order
#             pred = int(model.predict(X_input)[0])
#             # handle probability: if model has predict_proba
#             try:
#                 prob = float(model.predict_proba(X_input)[0][1] * 100)
#             except Exception:
#                 prob = 50.0  # fallback

#             # Heartbeat + result badges (UI features)
#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#             # Risk gauge (UI)
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob,2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range':[0,100]},
#                     'bar': {'color': ACCENT},
#                     'steps': [
#                         {'range':[0,33], 'color':'rgba(0,128,255,0.12)'},
#                         {'range':[33,66], 'color':'rgba(255,204,0,0.12)'},
#                         {'range':[66,100], 'color':'rgba(255,0,0,0.12)'}
#                     ]
#                 }
#             ))
#             fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#             # ---------------------------
#             # Unique Feature: What-if analysis (UI feature #12)
#             # ---------------------------
#             st.markdown("### 🔁 What-if Analysis (modify sliders to see hypothetical change)")
#             w_col1, w_col2 = st.columns(2)
#             with w_col1:
#                 w_bp = st.slider("Hypothetical Resting BP", 80, 200, resting_bp)
#                 w_chol = st.slider("Hypothetical Cholesterol", 100, 400, cholesterol)
#             with w_col2:
#                 w_maxhr = st.slider("Hypothetical Max HR", 60, 202, max_hr)
#                 w_fbs = st.selectbox("Hypothetical Fasting BS >120?", ["No","Yes"], index=0 if fasting_bs_input=="No" else 1)

#             w_sex = sex  # keep same
#             w_chest = chest_pain
#             w_exang = 1 if exercise_angina_input=="Yes" else 0
#             w_fbs_val = 1 if w_fbs=="Yes" else 0

#             w_df = pd.DataFrame({
#                 "Age":[age],
#                 "Sex":[w_sex],
#                 "ChestPainType":[w_chest],
#                 "RestingBP":[w_bp],
#                 "Cholesterol":[w_chol],
#                 "FastingBS":[w_fbs_val],
#                 "RestingECG":[resting_ecg],
#                 "MaxHR":[w_maxhr],
#                 "ExerciseAngina":[w_exang],
#                 "ST_Slope":[st_slope]
#             })

#             if model_source == "trained" and scaler is not None:
#                 w_in = scaler.transform(w_df)
#             else:
#                 w_in = w_df.values

#             try:
#                 w_prob = model.predict_proba(w_in)[0][1] * 100
#                 st.write(f"**Hypothetical risk:** {w_prob:.2f}%")
#             except Exception:
#                 st.write("Hypothetical risk not available (model has no predict_proba).")

#             # ---------------------------
#             # AI Doctor Advice generator (UI feature #11)
#             # ---------------------------
#             st.markdown("### 🩺 AI Health Advice (rule-based)")
#             adv = []
#             if cholesterol > 240:
#                 adv.append("Reduce saturated fats & consult nutritionist — high cholesterol.")
#             if resting_bp > 140:
#                 adv.append("Monitor salt intake, consult physician for BP control.")
#             if fasting_bs == 1:
#                 adv.append("Check blood sugar control — consider diabetes screening.")
#             if exercise_angina == 1:
#                 adv.append("Avoid strenuous exercise until cardiology evaluation.")
#             if max_hr < (220 - age) * 0.6:
#                 adv.append("Consider cardio conditioning with supervised training.")
#             if not adv:
#                 adv.append("Maintain healthy lifestyle: balanced diet, regular exercise, sleep 7-8 hours.")

#             for a in adv:
#                 st.markdown(f"- {a}")

#             # ---------------------------
#             # Feature importance (UI feature #6)
#             # ---------------------------
#             if model_source == "trained":
#                 # permutation importance on a small sample for speed
#                 sample_X = data_df.drop(columns=["target"]).sample(n=min(200, len(data_df)), random_state=1)
#                 sample_y = data_df.loc[sample_X.index, "target"]
#                 try:
#                     r = permutation_importance(model, scaler.transform(sample_X), sample_y, n_repeats=10, random_state=0)
#                     imp_df = pd.DataFrame({"feature": sample_X.columns, "importance": r.importances_mean})
#                     imp_df = imp_df.sort_values("importance", ascending=False).head(8)
#                     fig_imp = px.bar(imp_df, x="importance", y="feature", orientation='h', title="Top feature importances (permutation)")
#                     st.plotly_chart(fig_imp, use_container_width=True)
#                 except Exception:
#                     pass
#             else:
#                 st.markdown("<div class='small-muted'>Feature importance not shown for external model.</div>", unsafe_allow_html=True)

#             # ---------------------------
#             # Small Confusion Matrix preview (UI)
#             # ---------------------------
#             try:
#                 # quick evaluation on part of dataset for insight
#                 if model_source == "trained" and scaler is not None:
#                     X_all = data_df.drop(columns=["target"])
#                     y_all = data_df["target"]
#                     Xs = scaler.transform(X_all)
#                     y_pred_all = model.predict(Xs)
#                 else:
#                     X_all = data_df.drop(columns=["target"])
#                     y_all = data_df["target"]
#                     y_pred_all = model.predict(X_all.values)
#                 cm = confusion_matrix(y_all, y_pred_all)
#                 cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
#                 cm_fig.update_layout(title="Confusion Matrix (sample)", height=300, margin=dict(t=30))
#                 st.plotly_chart(cm_fig, use_container_width=True)
#             except Exception:
#                 pass

#         except Exception as e:
#             st.error(f"Prediction error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk predict (CSV) + file validation (UI feature #2)
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")
#     st.markdown("<div class='small-muted'>CSV must contain the 10 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, ST_Slope</div>", unsafe_allow_html=True)

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             expected_columns = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"]
#             if not set(expected_columns).issubset(df.columns):
#                 st.warning("Uploaded CSV doesn't have the required columns. Please follow the template.")
#             else:
#                 if model_source == "trained" and scaler is not None:
#                     Xb = scaler.transform(df[expected_columns])
#                 else:
#                     Xb = df[expected_columns].values
#                 preds = model.predict(Xb)
#                 try:
#                     probs = model.predict_proba(Xb)[:,1]*100
#                 except Exception:
#                     probs = np.zeros(len(preds))
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs,2)
#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error processing CSV: {e}")
#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model insights (UI features: correlation heatmap #7, risk histogram #8)
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy_map = {
#         "Decision Tree":80.97, 
#         "Logistic Regression":85.86, 
#         "Random Forest":84.23, 
#         "SVM":89.75
#     }
#     acc_df = pd.DataFrame(list(accuracy_map.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # accuracy bar with color depending on value (UI change requested earlier)
#     def color_for_acc(a):
#         if a >= 88:
#             return "green"
#         if a >= 83:
#             return "orange"
#         return "red"

#     colors = [color_for_acc(a) for a in acc_df["Accuracy"]]
#     fig = go.Figure(go.Bar(x=acc_df["Model"], y=acc_df["Accuracy"], marker_color=colors, hovertemplate='%{y:.2f}%<extra></extra>'))
#     fig.update_layout(yaxis=dict(range=[0,100]), margin=dict(t=10,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)")
#     st.plotly_chart(fig, use_container_width=True)

#     # Correlation heatmap (UI feature #7)
#     if "target" in data_df.columns:
#         corr = data_df.corr()
#         corr_fig = px.imshow(corr, text_auto=True, title="Feature Correlation (dataset)")
#         st.plotly_chart(corr_fig, use_container_width=True)

#     # Risk distribution histogram (UI feature #8)
#     try:
#         if model_source == "trained" and scaler is not None:
#             Xs = scaler.transform(data_df.drop(columns=["target"]))
#             probs_all = model.predict_proba(Xs)[:,1]
#         else:
#             Xs = data_df.drop(columns=["target"]).values
#             probs_all = model.predict_proba(Xs)[:,1]
#         hist_fig = px.histogram(probs_all*100, nbins=25, labels={"value":"Risk (%)"}, title="Predicted Risk Distribution in Dataset")
#         st.plotly_chart(hist_fig, use_container_width=True)
#     except Exception:
#         pass

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 4 — Extras: ECG sim (#14), advice, digital twin upgrade (#15)
# # ---------------------------
# with tab4:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Extras: ECG Simulator, Digital Twin & More")

#     # ECG Simulator (simple animated-like using Plotly)
#     st.markdown("### Animated ECG (simulator)")
#     t = np.linspace(0, 2*np.pi, 500)
#     # create synthetic ECG-like waveform
#     ecg = 0.6*np.sin(5*t) + 0.15*np.sin(50*t) + 0.03*np.random.randn(len(t))
#     ecg_fig = go.Figure()
#     ecg_fig.add_trace(go.Scatter(x=np.arange(len(ecg)), y=ecg, mode='lines', name='ECG'))
#     ecg_fig.update_layout(height=250, margin=dict(t=10,b=10), xaxis=dict(showticklabels=False))
#     st.plotly_chart(ecg_fig, use_container_width=True)

#     # Digital Twin Heart (upgrade)
#     st.markdown("### Heart Digital Twin")
#     # change color depending on last prediction probability (if exists)
#     last_prob = None
#     try:
#         last_prob = prob  # from earlier predict block if user ran it
#     except Exception:
#         last_prob = None

#     if last_prob is None:
#         st.markdown("<div style='text-align:center; padding:20px;'>Run a prediction to animate the heart color</div>", unsafe_allow_html=True)
#     else:
#         heart_color = "red" if last_prob>66 else ("orange" if last_prob>33 else "skyblue")
#         beat_speed = "1s" if last_prob>66 else ("1.2s" if last_prob>33 else "1.6s")
#         st.markdown(f"""
#             <div style="display:flex; justify-content:center; align-items:center;">
#               <div style="width:160px; height:140px; background:{heart_color}; transform:rotate(-45deg); border-radius:16px; box-shadow:0 8px 30px rgba(0,0,0,0.12); animation: beat {beat_speed} infinite;">
#               </div>
#             </div>
#             <style>@keyframes beat {{ 0%{{transform:scale(1) rotate(-45deg)}} 50%{{transform:scale(1.12) rotate(-45deg)}} 100%{{transform:scale(1) rotate(-45deg)}} }}</style>
#         """, unsafe_allow_html=True)

#     # AI Doctor Chat style note (UI feature #11 extension)
#     st.markdown("### Quick AI Health Tips")
#     st.markdown("- Eat a balanced diet rich in fiber and omega-3s.")
#     st.markdown("- Walk 30 minutes daily / do moderate exercise.")
#     st.markdown("- Reduce sodium and processed foods to manage BP.")
#     st.markdown("- Avoid smoking and monitor blood sugar.")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)








# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"           # very light sky tint (top)
# BG_BOTTOM = "#e9f7ff"        # very light sky tint (bottom)
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"           # iOS blue accent
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"             # dark-blue text
# SUBTEXT = "#54657A"          # muted subtext
# SHADOW = "rgba(10,132,255,0.08)"

# # 🔵 Deep Blue for Model Insights
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* subtle section title */
# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# /* iOS style primary button */
# div.stButton > button, button[kind="primary"] {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# /* heartbeat icon */
# .heartbeat {{
#     font-size: 42px;
#     line-height: 1;
#     display:block;
#     margin: 6px auto 0;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except Exception as e:
#         st.error(f"Model load error: {e}")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model is None:
#             st.warning("Model not loaded.")
#         else:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", 
#                         unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)

#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                 },
#             ))
#             fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
#                               paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if model:
#                 preds = model.predict(df.values)
#                 probs = model.predict_proba(df.values)[:, 1] * 100
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs, 2)

#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {
#         "Decision Tree":80.97, 
#         "Logistic Regression":85.86, 
#         "Random Forest":84.23, 
#         "SVM":89.75
#     }

#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # 🔵 Deep blue bar chart
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE),
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))

#     fig.update_layout(
#         yaxis=dict(range=[0,100]),
#         margin=dict(t=10, b=20, l=20, r=20),
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)






# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import plotly.express as px
# import base64
# import sqlite3
# import io
# import os
# import datetime
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Optional imports (SHAP)
# try:
#     import shap
#     _HAS_SHAP = True
# except Exception:
#     _HAS_SHAP = False

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — Full Feature App",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="💙"
# )

# # ---------------------------
# # Colors & palette (user palette from provided link)
# # ---------------------------
# PALETTE = {
#     "deep_blue": "#003A92",
#     "violet": "#8A2BE2",
#     "teal": "#20B2AA",
#     "aqua": "#7FDBFF",
#     "soft_pink": "#FF6F91",
#     "accent": "#0A84FF"
# }

# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.90)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"

# # ---------------------------
# # Styling
# # ---------------------------
# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}
# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 14px;
#     padding: 16px;
#     box-shadow: 0 6px 20px {SHADOW};
#     margin-bottom: 18px;
# }}
# .small-muted {{ color: {SUBTEXT}; font-size:13px; }}
# .badge-low {{ background:{PALETTE['aqua']}; color:#063; padding:8px; border-radius:8px; }}
# .badge-med {{ background:#FFD966; color:#4a3a00; padding:8px; border-radius:8px; }}
# .badge-high {{ background:#FF6B6B; color:white; padding:8px; border-radius:8px; }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Utilities: DB for history
# # ---------------------------
# DB_PATH = "predictions_history.db"

# def init_db():
#     conn = sqlite3.connect(DB_PATH, check_same_thread=False)
#     cur = conn.cursor()
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS history (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             ts TEXT,
#             inputs TEXT,
#             prediction INTEGER,
#             probability REAL,
#             advice TEXT
#         )
#     """)
#     conn.commit()
#     return conn

# conn = init_db()

# def save_history(inputs: dict, prediction: int, probability: float, advice_text: str):
#     cur = conn.cursor()
#     cur.execute("INSERT INTO history (ts, inputs, prediction, probability, advice) VALUES (?, ?, ?, ?, ?)",
#                 (datetime.datetime.utcnow().isoformat(), str(inputs), int(prediction), float(probability), advice_text))
#     conn.commit()

# def load_history(limit=200):
#     cur = conn.cursor()
#     cur.execute("SELECT id, ts, inputs, prediction, probability, advice FROM history ORDER BY id DESC LIMIT ?", (limit,))
#     rows = cur.fetchall()
#     cols = ["id", "timestamp", "inputs", "prediction", "probability", "advice"]
#     return pd.DataFrame(rows, columns=cols)

# # ---------------------------
# # Dataset loading (fallback)
# # ---------------------------
# @st.cache_data
# def load_dataset():
#     try:
#         df = pd.read_csv("heart.csv")
#     except Exception:
#         # remote fallback
#         df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/heart.csv")
#     return df

# data_df = load_dataset()

# # ---------------------------
# # Models: try load user model else train ensemble
# # ---------------------------
# @st.cache_resource
# def build_models():
#     models = {}
#     scaler = StandardScaler()
#     # Try user-supplied LogisticR.pkl like previous code
#     try:
#         user_model = pickle.load(open("LogisticR.pkl","rb"))
#         models["user"] = user_model
#         # no scaler for user model
#         return models, None, "loaded"
#     except Exception:
#         pass

#     # Train fallback models on dataset
#     df = data_df.copy()
#     if "target" in df.columns:
#         X = df.drop(columns=["target"])
#         y = df["target"]
#     else:
#         X = df.iloc[:, :-1]
#         y = df.iloc[:, -1]

#     # simple preprocessing: fillna
#     X = X.fillna(X.median(numeric_only=True))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler.fit(X_train)
#     X_train_s = scaler.transform(X_train)
#     X_test_s = scaler.transform(X_test)

#     lr = LogisticRegression(max_iter=1000)
#     rf = RandomForestClassifier(n_estimators=200, random_state=42)
#     svc = SVC(probability=True)

#     lr.fit(X_train_s, y_train)
#     rf.fit(X_train_s, y_train)
#     svc.fit(X_train_s, y_train)

#     models["LogisticRegression"] = lr
#     models["RandomForest"] = rf
#     models["SVC"] = svc

#     # compute basic accuracies to show
#     scores = {k: accuracy_score(y_test, models[k].predict(X_test_s)) for k in models}
#     return models, scaler, "trained"

# models, scaler, model_source = build_models()

# # ---------------------------
# # Helper: predict ensemble & single
# # ---------------------------
# def predict_models(input_df):
#     results = {}
#     if model_source == "trained" and scaler is not None:
#         X_in = scaler.transform(input_df)
#     else:
#         # if user model loaded, expect full features as values
#         X_in = input_df.values

#     for name, m in models.items():
#         try:
#             prob = float(m.predict_proba(X_in)[0][1])
#         except Exception:
#             # fallback if no predict_proba
#             try:
#                 pred = m.predict(X_in)[0]
#                 prob = float(pred)
#             except Exception:
#                 prob = 0.5
#         results[name] = prob
#     return results

# def ensemble_score(results_dict):
#     vals = list(results_dict.values())
#     return float(np.mean(vals))

# # ---------------------------
# # Sidebar: Patient profile + history quick access
# # ---------------------------
# st.sidebar.markdown("<div class='container-card'><strong>Patient Dashboard</strong></div>", unsafe_allow_html=True)
# st.sidebar.markdown("## Quick Actions")
# if st.sidebar.button("Show history (recent 20)"):
#     st.sidebar.write(load_history(20))

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights", "✨ Extras"])

# # ---------------------------
# # TAB 1 — Predict (Many features included)
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.header("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male","Female"])
#         chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)
#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No","Yes"])
#         resting_ecg = st.selectbox("Resting ECG", ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No","Yes"])
#         st_slope = st.selectbox("ST Slope", ["Upsloping","Flat","Downsloping"])

#     # patient summary
#     st.markdown('<div class="patient-summary">', unsafe_allow_html=True)
#     st.write(f"**Patient** — Age:{age}  Sex:{sex_input}  Chest:{chest_pain}")
#     st.write(f"RestingBP:{resting_bp}  Chol:{cholesterol}  MaxHR:{max_hr}")
#     st.markdown('</div>', unsafe_allow_html=True)

#     # encode
#     sex = 1 if sex_input=="Male" else 0
#     chest_idx = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain)
#     fasting_bs = 1 if fasting_bs_input=="Yes" else 0
#     rest_ecg_idx = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg)
#     exang = 1 if exercise_angina_input=="Yes" else 0
#     slope_idx = ["Upsloping","Flat","Downsloping"].index(st_slope)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_idx],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[rest_ecg_idx],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exang],
#         "ST_Slope":[slope_idx]
#     })

#     # Predict now
#     if st.button("Predict Now"):
#         try:
#             results = predict_models(input_df)
#             ensemble = ensemble_score(results)
#             prob_pct = ensemble * 100
#             # risk category
#             if prob_pct >= 66:
#                 badge_html = "<div class='badge-high'>High Risk</div>"
#                 risk_label = "High"
#             elif prob_pct >= 33:
#                 badge_html = "<div class='badge-med'>Moderate Risk</div>"
#                 risk_label = "Moderate"
#             else:
#                 badge_html = "<div class='badge-low'>Low Risk</div>"
#                 risk_label = "Low"
#             st.markdown(badge_html, unsafe_allow_html=True)

#             # store advice (rule-based)
#             advice = []
#             if cholesterol > 240: advice.append("High cholesterol — reduce saturated fats.")
#             if resting_bp > 140: advice.append("High BP — consult physician for BP control.")
#             if fasting_bs == 1: advice.append("Fasting BS >120 — consider diabetes screening.")
#             if exang == 1: advice.append("Exercise-induced angina present — cardiology review recommended.")
#             if not advice: advice.append("Maintain healthy lifestyle: diet, exercise, sleep.")

#             # save history
#             save_history(input_df.iloc[0].to_dict(), int(ensemble>=0.5), float(prob_pct), "; ".join(advice))

#             # ensemble display
#             st.metric("Ensemble Risk (%)", f"{prob_pct:.2f}%")
#             # show each model results with color palette
#             rows = []
#             for i,(k,v) in enumerate(results.items()):
#                 rows.append({"Model":k,"Prob (%)":round(v*100,2),"Color":list(PALETTE.values())[i % len(PALETTE)]})
#             res_df = pd.DataFrame(rows)
#             st.table(res_df[["Model","Prob (%)"]])

#             # risk gauge
#             gauge_col = PALETTE['deep_blue'] if prob_pct<33 else ("#FFD966" if prob_pct<66 else "#FF6B6B")
#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob_pct,2),
#                 title={'text':"Estimated Risk (%)"},
#                 gauge={'axis':{'range':[0,100]}, 'bar':{'color':gauge_col}}
#             ))
#             st.plotly_chart(fig, use_container_width=True)

#             # rule-based advice output
#             st.markdown("### Personalized Advice")
#             for a in advice:
#                 st.write("- " + a)

#             # What-if sliders (medication simulator)
#             st.markdown("### Medication / Intervention Simulator")
#             colA, colB = st.columns(2)
#             with colA:
#                 reduce_chol = st.slider("Simulate Cholesterol reduction (mg/dl)", 0, 100, 0)
#                 reduce_bp = st.slider("Simulate Resting BP reduction (mmHg)", 0, 50, 0)
#             # compute hypothetical
#             hypo = input_df.copy()
#             hypo.loc[0,"Cholesterol"] = max(0, hypo.loc[0,"Cholesterol"] - reduce_chol)
#             hypo.loc[0,"RestingBP"] = max(0, hypo.loc[0,"RestingBP"] - reduce_bp)
#             try:
#                 hypo_preds = predict_models(hypo)
#                 hypo_ensemble = ensemble_score(hypo_preds) * 100
#                 st.write(f"Hypothetical ensemble risk: **{hypo_ensemble:.2f}%**")
#             except Exception:
#                 st.write("Hypothetical prediction unavailable.")

#             # Explanation: SHAP or permutation importance
#             st.markdown("### Why this prediction? (Feature importance)")
#             if _HAS_SHAP and model_source=="trained":
#                 try:
#                     explainer = shap.Explainer(models["RandomForest"], scaler.transform(data_df.drop(columns=["target"])))
#                     shap_vals = explainer(scaler.transform(input_df))
#                     shap.plots.waterfall(shap_vals[0])
#                     st.pyplot(bbox_inches='tight')
#                 except Exception as e:
#                     st.write("SHAP failed; falling back to permutation importance.")
#                     _HAS_SHAP = False

#             if not _HAS_SHAP:
#                 try:
#                     sample_X = data_df.drop(columns=["target"]).sample(n=min(200, len(data_df)), random_state=1)
#                     sample_y = data_df.loc[sample_X.index, "target"]
#                     r = permutation_importance(models["RandomForest"], scaler.transform(sample_X), sample_y, n_repeats=10, random_state=1)
#                     imp_df = pd.DataFrame({"feature": sample_X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(8)
#                     fig_imp = px.bar(imp_df, x="importance", y="feature", orientation='h', title="Top features (permutation importance)")
#                     st.plotly_chart(fig_imp, use_container_width=True)
#                 except Exception:
#                     st.write("Feature importance not available.")

#             # small confusion matrix preview
#             try:
#                 if model_source=="trained":
#                     X_all = data_df.drop(columns=["target"])
#                     y_all = data_df["target"]
#                     ypred = models["RandomForest"].predict(scaler.transform(X_all))
#                     cm = confusion_matrix(y_all, ypred)
#                     cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
#                     cm_fig.update_layout(title="Confusion Matrix (RandomForest sample)", height=300)
#                     st.plotly_chart(cm_fig, use_container_width=True)
#             except Exception:
#                 pass

#             # Save downloadable HTML report (simple)
#             def generate_html_report(info):
#                 html = f"""
#                 <html><body>
#                 <h2>Heart Risk Report</h2>
#                 <p><b>Date:</b> {datetime.datetime.utcnow().isoformat()}</p>
#                 <h3>Inputs</h3><pre>{info['inputs']}</pre>
#                 <h3>Ensemble Risk</h3><p>{info['risk']:.2f}% ({risk_label})</p>
#                 <h3>Advice</h3><ul>{"".join([f"<li>{x}</li>" for x in info['advice']])}</ul>
#                 </body></html>
#                 """
#                 return html

#             report_info = {"inputs": input_df.to_dict(orient="records")[0], "risk": prob_pct, "advice": advice}
#             html_report = generate_html_report(report_info)
#             b = base64.b64encode(html_report.encode()).decode()
#             href = f'<a href="data:text/html;base64,{b}" download="heart_report.html">⬇ Download report (HTML)</a>'
#             st.markdown(href, unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"Prediction failed: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk Predict (CSV)
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.header("Batch Predictions (CSV)")

#     uploaded_file = st.file_uploader("Upload CSV with columns: Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,ST_Slope", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             expected = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","ST_Slope"]
#             if not set(expected).issubset(df.columns):
#                 st.warning("CSV missing required columns. Please follow template.")
#             else:
#                 Xb = df[expected]
#                 if model_source=="trained":
#                     Xb_s = scaler.transform(Xb)
#                 else:
#                     Xb_s = Xb.values
#                 preds = models["RandomForest"].predict(Xb_s) if "RandomForest" in models else list(models.values())[0].predict(Xb_s)
#                 try:
#                     probs = models["RandomForest"].predict_proba(Xb_s)[:,1]*100
#                 except Exception:
#                     probs = np.zeros(len(preds))
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs,2)
#                 st.success("Batch predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error reading CSV: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model Insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.header("Model Insights & Diagnostics")

#     # model accuracies (if trained)
#     if model_source=="trained":
#         # evaluate on test split quickly
#         df = data_df.copy()
#         X = df.drop(columns=["target"])
#         y = df["target"]
#         Xs = scaler.transform(X)
#         accs = {name: round(accuracy_score(y, m.predict(Xs))*100,2) for name,m in models.items() if name in models}
#     else:
#         accs = {"UserModel": 0.0}

#     acc_df = pd.DataFrame(list(accs.items()), columns=["Model","Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # Use user's palette to display bars (map models to palette entries)
#     colors = [PALETTE[k] if k in PALETTE else list(PALETTE.values())[i % len(PALETTE)] for i,k in enumerate(acc_df["Model"])]
#     # If not enough palette entries just cycle
#     if len(colors) < len(acc_df):
#         colors = colors + [list(PALETTE.values())[-1]] * (len(acc_df)-len(colors))

#     fig = go.Figure(go.Bar(x=acc_df["Model"], y=acc_df["Accuracy"], marker_color=colors, hovertemplate='%{y:.2f}%<extra></extra>'))
#     fig.update_layout(yaxis=dict(range=[0,100]), height=360)
#     st.plotly_chart(fig, use_container_width=True)

#     # correlation heatmap
#     try:
#         if "target" in data_df.columns:
#             corr = data_df.corr()
#             corr_fig = px.imshow(corr, text_auto=True, title="Feature Correlation (dataset)")
#             st.plotly_chart(corr_fig, use_container_width=True)
#     except Exception:
#         pass

#     # risk distribution histogram
#     try:
#         if model_source=="trained":
#             probs_all = models["RandomForest"].predict_proba(scaler.transform(data_df.drop(columns=["target"])))[:,1]
#             hist = px.histogram(probs_all*100, nbins=30, title="Predicted Risk Distribution (%)")
#             st.plotly_chart(hist, use_container_width=True)
#     except Exception:
#         pass

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 4 — Extras: ECG sim, Digital Twin, Symptom checker
# # ---------------------------
# with tab4:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.header("Extras & Tools")

#     # ECG simulator
#     st.markdown("### Animated ECG Simulator")
#     t = np.linspace(0, 2*np.pi, 400)
#     ecg = 0.6*np.sin(5*t) + 0.1*np.sin(50*t) + 0.02*np.random.randn(len(t))
#     ecg_fig = go.Figure()
#     ecg_fig.add_trace(go.Scatter(y=ecg, mode='lines'))
#     ecg_fig.update_layout(height=250, margin=dict(t=10,b=10), xaxis=dict(showticklabels=False))
#     st.plotly_chart(ecg_fig, use_container_width=True)

#     # ECG upload + basic arrhythmia detection
#     st.markdown("### Upload ECG CSV (single column of samples)")
#     ecg_file = st.file_uploader("Upload ECG CSV (optional)", type=["csv"])
#     if ecg_file:
#         try:
#             ecg_df = pd.read_csv(ecg_file, header=None)
#             samples = ecg_df.iloc[:,0].values
#             # naive detection: high variance or extreme peaks -> flag
#             peak_rate = np.mean(np.diff(np.where(samples > np.mean(samples)+2*np.std(samples))[0]))
#             st.write(f"Samples length: {len(samples)}")
#             # simple heuristic for arrhythmia (not clinical)
#             hr_est = 60 * (len(np.where(samples > np.mean(samples)+1.5*np.std(samples))[0]) / len(samples))
#             st.write(f"Estimated HR proxy: {hr_est:.1f}")
#             if np.std(samples) > 0.5:
#                 st.warning("ECG signal shows high variance — consider medical review.")
#             st.line_chart(samples)
#         except Exception as e:
#             st.error(f"Failed to read ECG: {e}")

#     # Digital twin (colored block)
#     st.markdown("### Heart Digital Twin")
#     last_p = None
#     try:
#         last_p = prob  # if earlier predicted
#     except Exception:
#         last_p = None
#     if last_p is None:
#         st.info("Run a prediction to animate the heart")
#     else:
#         color = "#FF6B6B" if last_p>66 else ("#FFD966" if last_p>33 else "#7FDBFF")
#         speed = "1s" if last_p>66 else ("1.2s" if last_p>33 else "1.6s")
#         st.markdown(f"""
#         <div style="display:flex; justify-content:center;">
#           <div style="width:140px; height:120px; background:{color}; transform:rotate(-45deg); border-radius:16px;
#                       box-shadow:0 8px 30px rgba(0,0,0,0.12); animation: beat {speed} infinite;"></div>
#         </div>
#         <style>@keyframes beat {{ 0%{{transform:scale(1) rotate(-45deg)}} 50%{{transform:scale(1.08) rotate(-45deg)}} 100%{{transform:scale(1) rotate(-45deg)}} }}</style>
#         """, unsafe_allow_html=True)

#     # Symptom checker (simple rule-based)
#     st.markdown("### Symptom Checker")
#     s_chest = st.checkbox("Chest pain")
#     s_breath = st.checkbox("Shortness of breath")
#     s_dizzy = st.checkbox("Dizziness / fainting")
#     s_palpit = st.checkbox("Palpitations")
#     if st.button("Check symptoms"):
#         score = sum([s_chest, s_breath, s_dizzy, s_palpit])
#         if s_chest or (s_breath and s_dizzy):
#             st.warning("Symptoms may be cardiac-related. Seek urgent medical advice.")
#         elif score >= 2:
#             st.info("Symptoms moderate — consider clinical checkup.")
#         else:
#             st.success("Symptoms low-risk — monitor and follow healthy lifestyle.")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* iOS primary button */
# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # CSV Download Helper
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'

# # ---------------------------
# # Health Suggestion Generator
# # ---------------------------
# def get_health_suggestions(row):
#     suggestions = []

#     if row["RestingBP"] > 130:
#         suggestions.append("Reduce salt intake and check blood pressure regularly.")
#     if row["Cholesterol"] > 240:
#         suggestions.append("Cholesterol high — avoid oily foods and choose fiber-rich diet.")
#     if row["MaxHR"] < 120:
#         suggestions.append("Low maximum heart rate — perform regular light cardio.")
#     if row["FastingBS"] == 1:
#         suggestions.append("High fasting sugar — reduce sugar and avoid white rice.")
#     if row["ExerciseAngina"] == 1:
#         suggestions.append("Avoid heavy exercise until angina is evaluated by a doctor.")
#     if row["RestingECG"] != 0:
#         suggestions.append("ECG abnormality detected — follow up with a cardiologist.")
#     if row["ST_Slope"] == 1:
#         suggestions.append("Flat ST slope may indicate reduced blood flow to heart.")
#     if row["ST_Slope"] == 2:
#         suggestions.append("Downsloping ST slope — seek medical attention soon.")
#     if row["ChestPainType"] == 0:
#         suggestions.append("Typical Angina — could indicate coronary blockage.")
#     if row["ChestPainType"] == 3:
#         suggestions.append("Asymptomatic — regular screening recommended.")

#     return suggestions[:5]

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction System</h1>
# <div style="text-align:center; color:#54657A;">AI-powered Health Insights</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # Load Model
# # ---------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except:
#     model = None
#     st.error("Model not found. Ensure LogisticR.pkl is in the same folder.")

# # ---------------------------
# # TAB 1 — Single Predict
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs_input == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fast],
#         "RestingECG":[ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[angina],
#         "ST_Slope":[slope]
#     })

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Gauge Chart
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(prob,2),
#             title={'text': "Risk Probability (%)"},
#             gauge={'axis': {'range': [0,100]}, 'bar': {'color': ACCENT}}
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         # Suggestions
#         tips = get_health_suggestions(input_df.iloc[0])
#         st.markdown("### 💡 Personalized Health Suggestions")
#         for t in tips:
#             st.markdown(f"- {t}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk Predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     file = st.file_uploader("Upload CSV", type=["csv"])
#     if file:
#         data = pd.read_csv(file)
#         preds = model.predict(data)
#         probs = model.predict_proba(data)[:,1] * 100

#         data["Prediction"] = preds
#         data["Probability"] = np.round(probs,2)
#         data["Suggestions"] = data.apply(lambda x: "; ".join(get_health_suggestions(x)), axis=1)

#         st.success("Predictions completed.")
#         st.dataframe(data)

#         st.markdown(get_csv_download_link(data), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model Insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Comparison")

#     accuracy = {
#         "Decision Tree":80.97,
#         "Logistic Regression":85.86,
#         "Random Forest":84.23,
#         "SVM":89.75
#     }

#     df = pd.DataFrame(accuracy.items(), columns=["Model","Accuracy"])
#     st.dataframe(df)

#     fig = go.Figure(go.Bar(
#         x=df["Model"],
#         y=df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE)
#     ))
#     fig.update_layout(yaxis=dict(range=[0,100]))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div style='text-align:center; color:#666; padding:10px;'>
# 🔍 Educational demo — Not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)




# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME (single theme)
# # ---------------------------
# BG_TOP = "#f6fbff"           # very light sky tint (top)
# BG_BOTTOM = "#e9f7ff"        # very light sky tint (bottom)
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"           # iOS blue accent
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"             # dark-blue text
# SUBTEXT = "#54657A"          # muted subtext
# SHADOW = "rgba(10,132,255,0.08)"

# # 🔵 Deep Blue for Model Insights
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* subtle section title */
# .section-subtitle {{
#     color: {SUBTEXT};
#     font-size: 14px;
#     margin-top: -6px;
#     margin-bottom: 10px;
# }}

# /* iOS style primary button */
# div.stButton > button, button[kind="primary"] {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}

# /* heartbeat icon */
# .heartbeat {{
#     font-size: 42px;
#     line-height: 1;
#     display:block;
#     margin: 6px auto 0;
#     animation: pulse 1s infinite;
# }}
# @keyframes pulse {{
#     0% {{ transform: scale(1); opacity: 0.95; }}
#     50% {{ transform: scale(1.08); opacity: 1; }}
#     100% {{ transform: scale(1); opacity: 0.95; }}
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # Helper: download CSV
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a class="stDownloadButton" href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
#     return href

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <div style="display:flex; align-items:center; gap:14px;">
#   <div style="background:linear-gradient(180deg, rgba(10,132,255,0.15), rgba(10,132,255,0.06)); 
#               width:54px; height:54px; border-radius:13px; display:flex; align-items:center; justify-content:center;">
#     <span style="font-size:26px;">💙</span>
#   </div>
#   <div>
#     <h1 style="margin:0; font-size:22px;">Advanced Heart Disease Prediction</h1>
#     <div class="section-subtitle">iOS-style Sky-Blue UI • Clean & minimal</div>
#   </div>
# </div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # TAB 1 — Single prediction
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns([1,1])
#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise-induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode
#     sex = 1 if sex_input == "Male" else 0
#     chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_input)
#     fasting_bs = 1 if fasting_bs_input == "Yes" else 0
#     resting_ecg = ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     exercise_angina = 1 if exercise_angina_input == "Yes" else 0
#     st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest_pain],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fasting_bs],
#         "RestingECG":[resting_ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[exercise_angina],
#         "ST_Slope":[st_slope]
#     })

#     # Load model
#     try:
#         model = pickle.load(open("LogisticR.pkl", "rb"))
#     except Exception as e:
#         st.error(f"Model load error: {e}")
#         model = None

#     if st.button("💙 Predict Now"):
#         if model is None:
#             st.warning("Model not loaded.")
#         else:
#             pred = int(model.predict(input_df.values)[0])
#             prob = float(model.predict_proba(input_df.values)[0][1] * 100)

#             st.markdown("<div style='text-align:center;'><span class='heartbeat'>❤️</span></div>", 
#                         unsafe_allow_html=True)

#             if pred == 1:
#                 st.markdown("<div class='result-positive'>❤️ Risk of Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)
#             else:
#                 st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", 
#                             unsafe_allow_html=True)

#             fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=round(prob, 2),
#                 title={'text': "Risk Probability (%)"},
#                 gauge={
#                     'axis': {'range': [0, 100]},
#                     'bar': {'color': ACCENT},
#                     'bgcolor': "white",
#                 },
#             ))
#             fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), 
#                               paper_bgcolor="rgba(0,0,0,0)")
#             st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
#     if uploaded_file:
#         try:
#             df = pd.read_csv(uploaded_file)
#             if model:
#                 preds = model.predict(df.values)
#                 probs = model.predict_proba(df.values)[:, 1] * 100
#                 df_out = df.copy()
#                 df_out["Prediction"] = preds
#                 df_out["Probability"] = np.round(probs, 2)

#                 st.success("Predictions completed")
#                 st.dataframe(df_out, use_container_width=True)
#                 st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)
#         except Exception as e:
#             st.error(f"Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy & Comparison")

#     accuracy = {
#         "Decision Tree":80.97, 
#         "Logistic Regression":85.86, 
#         "Random Forest":84.23, 
#         "SVM":89.75
#     }

#     acc_df = pd.DataFrame(list(accuracy.items()), columns=["Model", "Accuracy"])
#     st.dataframe(acc_df, use_container_width=True)

#     # 🔵 Deep blue bar chart
#     fig = go.Figure(go.Bar(
#         x=acc_df["Model"],
#         y=acc_df["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE),
#         hovertemplate='%{y:.2f}%<extra></extra>'
#     ))

#     fig.update_layout(
#         yaxis=dict(range=[0,100]),
#         margin=dict(t=10, b=20, l=20, r=20),
#         paper_bgcolor="rgba(0,0,0,0)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style="text-align:center; padding-top:12px; color:#6b7785; font-size:13px;">
#   This tool is for educational/demo purposes only — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)












# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # -----------------------------------------------------------
# # THEME COLORS
# # -----------------------------------------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# # -----------------------------------------------------------
# # CSS STYLE
# # -----------------------------------------------------------
# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}
# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
#   color: {TEXT};
# }}
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}
# .result-positive {{
#     background: #ffeaea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: #eaffea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # CSV DOWNLOAD HELPER
# # -----------------------------------------------------------
# def get_csv_download_link(df, filename="predictions.csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download CSV</a>'

# # -----------------------------------------------------------
# # HEALTH SUGGESTION ENGINE
# # -----------------------------------------------------------
# def get_health_suggestions(row):
#     tips = []

#     if row["RestingBP"] > 130:
#         tips.append("High BP — reduce salt and monitor weekly.")
#     if row["Cholesterol"] > 240:
#         tips.append("High cholesterol — avoid oily & fried foods.")
#     if row["FastingBS"] == 1:
#         tips.append("High blood sugar — reduce sugar and rice.")
#     if row["MaxHR"] < 120:
#         tips.append("Low Max HR — walk or do light cardio daily.")
#     if row["ExerciseAngina"] == 1:
#         tips.append("Exercise angina — avoid heavy workouts.")
#     if row["RestingECG"] != 0:
#         tips.append("ECG abnormality — follow-up with cardiologist.")
#     if row["ST_Slope"] == 2:
#         tips.append("Downsloping ST — immediate medical evaluation advised.")
#     if row["ChestPainType"] == 0:
#         tips.append("Typical Angina — possible coronary blockage.")

#     return tips[:5]

# # -----------------------------------------------------------
# # HEADER
# # -----------------------------------------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction</h1>
# <div style="text-align:center; color:#54657A;">AI-Powered Health Assessment</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TABS
# # -----------------------------------------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # -----------------------------------------------------------
# # LOAD MODEL
# # -----------------------------------------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except:
#     st.error("Model file (LogisticR.pkl) not found!")
#     model = None

# # -----------------------------------------------------------
# # TAB 1 — SINGLE PREDICTION
# # -----------------------------------------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Health Check")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type", 
#                 ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG", 
#                 ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
#         exercise_angina_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encoding
#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs_input == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age], "Sex":[sex], "ChestPainType":[chest],
#         "RestingBP":[resting_bp], "Cholesterol":[cholesterol],
#         "FastingBS":[fast], "RestingECG":[ecg], "MaxHR":[max_hr],
#         "ExerciseAngina":[angina], "ST_Slope":[slope]
#     })

#     if st.button("💙 Predict Now") and model is not None:
#         pred = model.predict(input_df)[0]
#         prob = float(model.predict_proba(input_df)[0][1] * 100)

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Gauge Chart
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=round(prob, 2),
#             title={'text': "Risk Probability (%)"},
#             gauge={'axis': {'range': [0, 100]}, 'bar': {'color': ACCENT}}
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         # Suggestions
#         st.subheader("💡 Personalized Health Suggestions")
#         for tip in get_health_suggestions(input_df.iloc[0]):
#             st.write(f"- {tip}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TAB 2 — BULK PREDICTION (CSV + EXCEL)
# # -----------------------------------------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV or Excel File")

#     uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

#     # Encoding maps
#     sex_map = {"Male":1, "Female":0}
#     chest_map = {"Typical Angina":0,"Atypical Angina":1,"Non-Anginal Pain":2,"Asymptomatic":3}
#     ecg_map = {"Normal":0,"ST-T wave Abnormality":1,"Left Ventricular Hypertrophy":2}
#     angina_map = {"No":0,"Yes":1}
#     slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}

#     if uploaded_file:
#         try:
#             # Read file
#             if uploaded_file.name.endswith(".csv"):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file)

#             # Auto-encode text values
#             for col, mapping in {
#                 "Sex":sex_map, "ChestPainType":chest_map,
#                 "RestingECG":ecg_map, "ExerciseAngina":angina_map,
#                 "ST_Slope":slope_map
#             }.items():
#                 if df[col].dtype == object:
#                     df[col] = df[col].map(mapping)

#             # Predict
#             preds = model.predict(df)
#             probs = model.predict_proba(df)[:, 1] * 100

#             df_out = df.copy()
#             df_out["Prediction"] = preds
#             df_out["Probability"] = np.round(probs, 2)

#             # Suggestions
#             df_out["Suggestions"] = df_out.apply(lambda r: "; ".join(get_health_suggestions(r)), axis=1)

#             st.success("Predictions completed")
#             st.dataframe(df_out)

#             # Download link
#             st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"❌ Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TAB 3 — MODEL INSIGHTS
# # -----------------------------------------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Performance Comparison")

#     accuracy = {
#         "Decision Tree":80.97,
#         "Logistic Regression":85.86,
#         "Random Forest":84.23,
#         "SVM":89.75
#     }

#     df_acc = pd.DataFrame(accuracy.items(), columns=["Model", "Accuracy"])
#     st.dataframe(df_acc)

#     fig = go.Figure(go.Bar(
#         x=df_acc["Model"], y=df_acc["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE)
#     ))
#     fig.update_layout(yaxis=dict(range=[0, 100]))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # FOOTER
# # -----------------------------------------------------------
# st.markdown("""
# <div style='text-align:center; padding:10px; color:#666;'>
# This tool is for educational use — not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)






# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"  # deep blue

# st.markdown(f"""
# <style>
# /* page background gradient */
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# /* base font */
# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# /* header */
# header .decoration {{
#   display: none;
# }}
# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# /* glass card */
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# /* iOS primary button */
# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# /* result badges */
# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # CSV Download Helper
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'

# # ---------------------------
# # Health Suggestion Generator
# # ---------------------------
# def get_health_suggestions(row):
#     suggestions = []

#     if row["RestingBP"] > 130:
#         suggestions.append("Reduce salt intake and check blood pressure regularly.")
#     if row["Cholesterol"] > 240:
#         suggestions.append("Cholesterol high — avoid oily foods and choose fiber-rich diet.")
#     if row["MaxHR"] < 120:
#         suggestions.append("Low maximum heart rate — perform regular light cardio.")
#     if row["FastingBS"] == 1:
#         suggestions.append("High fasting sugar — reduce sugar and avoid white rice.")
#     if row["ExerciseAngina"] == 1:
#         suggestions.append("Avoid heavy exercise until angina is evaluated by a doctor.")
#     if row["RestingECG"] != 0:
#         suggestions.append("ECG abnormality detected — follow up with a cardiologist.")
#     if row["ST_Slope"] == 1:
#         suggestions.append("Flat ST slope may indicate reduced blood flow to heart.")
#     if row["ST_Slope"] == 2:
#         suggestions.append("Downsloping ST slope — seek medical attention soon.")
#     if row["ChestPainType"] == 0:
#         suggestions.append("Typical Angina — could indicate coronary blockage.")
#     if row["ChestPainType"] == 3:
#         suggestions.append("Asymptomatic — regular screening recommended.")

#     return suggestions[:5]

# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction System</h1>
# <div style="text-align:center; color:#54657A;">AI-powered Health Insights</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # ---------------------------
# # Load Model
# # ---------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except:
#     model = None
#     st.error("Model not found. Ensure LogisticR.pkl is in the same folder.")

# # ---------------------------
# # TAB 1 — Single Predict
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs_input == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fast],
#         "RestingECG":[ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[angina],
#         "ST_Slope":[slope]
#     })

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Gauge Chart
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(prob,2),
#             title={'text': "Risk Probability (%)"},
#             gauge={'axis': {'range': [0,100]}, 'bar': {'color': ACCENT}}
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         # Suggestions
#         tips = get_health_suggestions(input_df.iloc[0])
#         st.markdown("### 💡 Personalized Health Suggestions")
#         for t in tips:
#             st.markdown(f"- {t}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 2 — Bulk Predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     file = st.file_uploader("Upload CSV", type=["csv"])
#     if file:
#         data = pd.read_csv(file)
#         preds = model.predict(data)
#         probs = model.predict_proba(data)[:,1] * 100

#         data["Prediction"] = preds
#         data["Probability"] = np.round(probs,2)
#         data["Suggestions"] = data.apply(lambda x: "; ".join(get_health_suggestions(x)), axis=1)

#         st.success("Predictions completed.")
#         st.dataframe(data)

#         st.markdown(get_csv_download_link(data), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------------
# # TAB 3 — Model Insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Comparison (Blue Palette 🎨)")

#     # Accuracy Values
#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     df = pd.DataFrame(accuracy.items(), columns=["Model","Accuracy"])
#     st.dataframe(df)

#     # BLUE PALETTE COLORS
#     blue_palette = [
#         "#89CFF0",  # baby blue
#         "#0A84FF",  # iOS accent blue
#         "#0074E8",  # royal blue
#         "#003A92"   # deep navy blue
#     ]

#     fig = go.Figure(go.Bar(
#         x=df["Model"],
#         y=df["Accuracy"],
#         marker=dict(color=blue_palette)
#     ))
#     fig.update_layout(
#         yaxis=dict(range=[0,100]),
#         title="Model Accuracy Comparison (Blue Theme)",
#         title_x=0.5
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)

# # Footer
# st.markdown("""
# <div style='text-align:center; color:#666; padding:10px;'>
# 🔍 Educational demo — Not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)















# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # ---------------------------
# # Page config
# # ---------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # ---------------------------
# # iOS-Style Sky-Blue THEME
# # ---------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# ACCENT_SOFT = "rgba(10,132,255,0.12)"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}

# html, body, [class*="css"]  {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
#   color: {TEXT};
# }}

# header .decoration {{ display: none; }}

# h1, h2, h3, h4 {{
#   color: {TEXT} !important;
#   font-weight: 600;
# }}

# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}

# div.stButton > button {{
#     background: linear-gradient(180deg, {ACCENT} 0%, #0074E8 100%);
#     color: white;
#     border-radius: 14px;
#     padding: 10px 18px;
#     font-size: 15px;
#     font-weight: 600;
#     box-shadow: 0 6px 18px rgba(10,132,255,0.18);
#     border: none;
#     transition: transform 0.12s ease, box-shadow 0.12s ease;
# }}
# div.stButton > button:hover {{
#     transform: translateY(-2px);
#     box-shadow: 0 10px 28px rgba(10,132,255,0.22);
# }}

# .result-positive {{
#     background: linear-gradient(90deg, rgba(255,242,242,0.9), rgba(255,238,238,0.8));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: linear-gradient(90deg, rgba(240,255,245,0.9), rgba(232,255,240,0.9));
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 4px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)


# # ---------------------------
# # CSV Download Helper
# # ---------------------------
# def get_csv_download_link(df, filename="predictions.csv", label="⬇ Download CSV"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'


# # ---------------------------
# # Health Suggestions
# # ---------------------------
# def get_health_suggestions(row):
#     suggestions = []

#     if row["RestingBP"] > 130:
#         suggestions.append("Reduce salt intake and monitor blood pressure regularly.")
#     if row["Cholesterol"] > 240:
#         suggestions.append("Avoid fried/oily foods and increase fiber intake.")
#     if row["MaxHR"] < 120:
#         suggestions.append("Low maximum heart rate — consider light daily cardio.")
#     if row["FastingBS"] == 1:
#         suggestions.append("Reduce sugar and avoid refined carbs like white rice.")
#     if row["ExerciseAngina"] == 1:
#         suggestions.append("Avoid heavy exercise until evaluated by a doctor.")
#     if row["RestingECG"] != 0:
#         suggestions.append("ECG abnormal — follow up with a cardiologist.")
#     if row["ST_Slope"] == 1:
#         suggestions.append("Flat ST slope suggests reduced blood flow — monitor symptoms.")
#     if row["ST_Slope"] == 2:
#         suggestions.append("Downsloping ST — seek medical attention soon.")
#     if row["ChestPainType"] == 0:
#         suggestions.append("Typical Angina — possible coronary blockage.")
#     if row["ChestPainType"] == 3:
#         suggestions.append("Asymptomatic — regular screening is recommended.")

#     return suggestions[:5]


# # ---------------------------
# # Header
# # ---------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction System</h1>
# <div style="text-align:center; color:#54657A;">AI-powered Health Insights</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # Tabs
# # ---------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # ---------------------------
# # Load Model
# # ---------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except:
#     model = None
#     st.error("Model file not found. Please include LogisticR.pkl")


# # ---------------------------
# # TAB 1 — Single Predict
# # ---------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs_input = st.selectbox("Fasting Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 202, 150)
#         exercise_angina_input = st.selectbox("Exercise Angina?", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs_input == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age],
#         "Sex":[sex],
#         "ChestPainType":[chest],
#         "RestingBP":[resting_bp],
#         "Cholesterol":[cholesterol],
#         "FastingBS":[fast],
#         "RestingECG":[ecg],
#         "MaxHR":[max_hr],
#         "ExerciseAngina":[angina],
#         "ST_Slope":[slope]
#     })

#     if st.button("💙 Predict Now"):
#         pred = model.predict(input_df)[0]
#         prob = model.predict_proba(input_df)[0][1] * 100

#         # ---------------------------
#         # DYNAMIC COLOR FOR GAUGE
#         # ---------------------------
#         if prob > 70:
#             gauge_color = "#FF3B30"      # RED
#         elif prob >= 40:
#             gauge_color = "#FFD60A"      # YELLOW
#         else:
#             gauge_color = "#0A84FF"      # BLUE

#         # Prediction result
#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Gauge Chart
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(prob, 2),
#             title={'text': "Risk Probability (%)"},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': gauge_color}
#             }
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         # Suggestions
#         tips = get_health_suggestions(input_df.iloc[0])
#         st.markdown("### 💡 Personalized Health Suggestions")
#         for t in tips:
#             st.markdown(f"- {t}")

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 2 — Bulk Predict
# # ---------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV for Batch Predictions")

#     file = st.file_uploader("Upload CSV", type=["csv"])
#     if file:
#         data = pd.read_csv(file)
#         preds = model.predict(data)
#         probs = model.predict_proba(data)[:,1] * 100

#         data["Prediction"] = preds
#         data["Probability"] = np.round(probs, 2)
#         data["Suggestions"] = data.apply(lambda x: "; ".join(get_health_suggestions(x)), axis=1)

#         st.success("Predictions completed.")
#         st.dataframe(data)

#         st.markdown(get_csv_download_link(data), unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # TAB 3 — Model Insights
# # ---------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Comparison (Blue Palette 🎨)")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     df = pd.DataFrame(accuracy.items(), columns=["Model","Accuracy"])
#     st.dataframe(df)

#     blue_palette = ["#89CFF0", "#0A84FF", "#0074E8", "#003A92"]

#     fig = go.Figure(go.Bar(
#         x=df["Model"],
#         y=df["Accuracy"],
#         marker=dict(color=blue_palette)
#     ))
#     fig.update_layout(yaxis=dict(range=[0, 100]), title="Model Accuracy Comparison", title_x=0.5)
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# # ---------------------------
# # Footer
# # ---------------------------
# st.markdown("""
# <div style='text-align:center; color:#666; padding:10px;'>
# 🔍 Educational demo — Not a medical diagnosis.
# </div>
# """, unsafe_allow_html=True)





# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # -----------------------------------------------------------
# # THEME COLORS
# # -----------------------------------------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# # -----------------------------------------------------------
# # CSS STYLE
# # -----------------------------------------------------------
# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}
# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
#   color: {TEXT};
# }}
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}
# .result-positive {{
#     background: #ffeaea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: #eaffea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)

# # -----------------------------------------------------------
# # DOWNLOAD CSV HELPER
# # -----------------------------------------------------------
# def get_csv_download_link(df, filename="predictions.csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download CSV</a>'


# # -----------------------------------------------------------
# # HEALTH SUGGESTION ENGINE
# # -----------------------------------------------------------
# def get_health_suggestions(row):
#     tips = []

#     if row["RestingBP"] > 130:
#         tips.append("High BP — reduce salt and monitor weekly.")
#     if row["Cholesterol"] > 240:
#         tips.append("High cholesterol — avoid oily & fried foods.")
#     if row["FastingBS"] == 1:
#         tips.append("High blood sugar — reduce sugar and rice.")
#     if row["MaxHR"] < 120:
#         tips.append("Low Max HR — walk or do light cardio daily.")
#     if row["ExerciseAngina"] == 1:
#         tips.append("Exercise-induced angina — avoid heavy workouts.")
#     if row["RestingECG"] != 0:
#         tips.append("ECG abnormality detected — follow up with cardiologist.")
#     if row["ST_Slope"] == 2:
#         tips.append("Downsloping ST — seek medical attention soon.")
#     if row["ChestPainType"] == 0:
#         tips.append("Typical Angina — possible coronary blockage.")

#     return tips[:5]


# # -----------------------------------------------------------
# # HEADER
# # -----------------------------------------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction</h1>
# <div style="text-align:center; color:#54657A;">AI-Powered Health Analysis</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TABS
# # -----------------------------------------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])

# # -----------------------------------------------------------
# # LOAD MODEL
# # -----------------------------------------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except Exception as e:
#     st.error("❌ Model file not found. Ensure 'LogisticR.pkl' is in the same folder.")
#     model = None


# # -----------------------------------------------------------
# # TAB 1 — SINGLE PREDICTION
# # -----------------------------------------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
#         exercise_angina_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encoding
#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age], "Sex":[sex], "ChestPainType":[chest],
#         "RestingBP":[resting_bp], "Cholesterol":[cholesterol],
#         "FastingBS":[fast], "RestingECG":[ecg], "MaxHR":[max_hr],
#         "ExerciseAngina":[angina], "ST_Slope":[slope]
#     })

#     if st.button("💙 Predict Now") and model is not None:
#         pred = model.predict(input_df)[0]
#         prob = float(model.predict_proba(input_df)[0][1] * 100)

#         if pred == 1:
#             st.markdown("<div class='result-positive'>❤️ Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'>💚 No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # Gauge Chart
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number", value=round(prob, 2),
#             title={'text': "Risk Probability (%)"},
#             gauge={'axis': {'range': [0, 100]}, 'bar': {'color': ACCENT}}
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("💡 Health Suggestions")
#         for tip in get_health_suggestions(input_df.iloc[0]):
#             st.write(f"- {tip}")

#     st.markdown('</div>', unsafe_allow_html=True)


# # -----------------------------------------------------------
# # TAB 2 — BULK PREDICT (CSV + EXCEL)
# # -----------------------------------------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV or Excel File")

#     uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

#     # Category → numeric mappings
#     sex_map = {"Male": 1, "Female": 0}
#     chest_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
#     ecg_map = {"Normal": 0, "ST-T wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
#     angina_map = {"No": 0, "Yes": 1}
#     slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

#     if uploaded_file:
#         try:
#             # Read file
#             if uploaded_file.name.endswith(".csv"):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file, engine="openpyxl")

#             # Apply encoding on text columns
#             mapping_dict = {
#                 "Sex": sex_map,
#                 "ChestPainType": chest_map,
#                 "RestingECG": ecg_map,
#                 "ExerciseAngina": angina_map,
#                 "ST_Slope": slope_map
#             }

#             for col, mapping in mapping_dict.items():
#                 if col in df.columns and df[col].dtype == object:
#                     df[col] = df[col].map(mapping)

#             # Predict
#             preds = model.predict(df)
#             probs = model.predict_proba(df)[:, 1] * 100

#             df_out = df.copy()
#             df_out["Prediction"] = preds
#             df_out["Probability"] = np.round(probs, 2)
#             df_out["Suggestions"] = df_out.apply(lambda r: "; ".join(get_health_suggestions(r)), axis=1)

#             st.success("Predictions completed!")
#             st.dataframe(df_out)

#             st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"❌ Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TAB 3 — MODEL INSIGHTS
# # -----------------------------------------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Comparison")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     df_acc = pd.DataFrame(accuracy.items(), columns=["Model", "Accuracy"])
#     st.dataframe(df_acc)

#     fig = go.Figure(go.Bar(
#         x=df_acc["Model"], y=df_acc["Accuracy"],
#         marker=dict(color=INSIGHT_BLUE)
#     ))

#     fig.update_layout(yaxis=dict(range=[0, 100]))
#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import plotly.graph_objects as go
# import base64

# # -----------------------------------------------------------
# # PAGE CONFIG
# # -----------------------------------------------------------
# st.set_page_config(
#     page_title="Heart Health — iOS Sky UI",
#     layout="centered",
#     initial_sidebar_state="collapsed",
#     page_icon="💙"
# )

# # -----------------------------------------------------------
# # THEME COLORS
# # -----------------------------------------------------------
# BG_TOP = "#f6fbff"
# BG_BOTTOM = "#e9f7ff"
# CARD_BG = "rgba(255,255,255,0.85)"
# ACCENT = "#0A84FF"
# TEXT = "#0B2545"
# SUBTEXT = "#54657A"
# SHADOW = "rgba(10,132,255,0.08)"
# INSIGHT_BLUE = "#003A92"

# # -----------------------------------------------------------
# # CSS STYLE
# # -----------------------------------------------------------
# st.markdown(f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
#   background: linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
#   min-height: 100vh;
# }}
# html, body, [class*="css"] {{
#   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
#   color: {TEXT};
# }}
# .container-card {{
#     background: {CARD_BG};
#     border-radius: 18px;
#     padding: 22px;
#     box-shadow: 0 6px 24px {SHADOW};
#     border: 1px solid rgba(255,255,255,0.6);
#     margin-bottom: 20px;
#     backdrop-filter: blur(8px);
# }}
# .result-positive {{
#     background: #ffeaea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #FF3B30;
#     font-weight: 600;
# }}
# .result-negative {{
#     background: #eaffea;
#     padding: 12px;
#     border-radius: 12px;
#     border-left: 5px solid #34C759;
#     font-weight: 600;
# }}
# </style>
# """, unsafe_allow_html=True)


# # -----------------------------------------------------------
# # DOWNLOAD CSV HELPER
# # -----------------------------------------------------------
# def get_csv_download_link(df, filename="predictions.csv"):
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download CSV</a>'


# # -----------------------------------------------------------
# # HEALTH SUGGESTION ENGINE
# # -----------------------------------------------------------
# def get_health_suggestions(row):
#     tips = []

#     if row["RestingBP"] > 130:
#         tips.append("High BP — reduce salt and monitor weekly.")
#     if row["Cholesterol"] > 240:
#         tips.append("High cholesterol — avoid oily & fried foods.")
#     if row["FastingBS"] == 1:
#         tips.append("High blood sugar — reduce sugar and rice.")
#     if row["MaxHR"] < 120:
#         tips.append("Low Max HR — walk or do light cardio daily.")
#     if row["ExerciseAngina"] == 1:
#         tips.append("Exercise-induced angina — avoid heavy workouts.")
#     if row["RestingECG"] != 0:
#         tips.append("ECG abnormality detected — follow up with cardiologist.")
#     if row["ST_Slope"] == 2:
#         tips.append("Downsloping ST — seek medical attention soon.")
#     if row["ChestPainType"] == 0:
#         tips.append("Typical Angina — possible coronary blockage.")

#     return tips[:5]


# # -----------------------------------------------------------
# # HEADER UI
# # -----------------------------------------------------------
# st.markdown('<div class="container-card">', unsafe_allow_html=True)
# st.markdown("""
# <h1 style="text-align:center;">💙 Heart Disease Prediction</h1>
# <div style="text-align:center; color:#54657A;">AI-Powered Health Analysis</div>
# """, unsafe_allow_html=True)
# st.markdown('</div>', unsafe_allow_html=True)


# # -----------------------------------------------------------
# # TABS
# # -----------------------------------------------------------
# tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# # -----------------------------------------------------------
# # LOAD MODEL
# # -----------------------------------------------------------
# try:
#     model = pickle.load(open("LogisticR.pkl", "rb"))
# except:
#     model = None
#     st.error("❌ LogisticR.pkl not found in folder.")


# # -----------------------------------------------------------
# # GAUGE COLOR FUNCTION
# # -----------------------------------------------------------
# def get_gauge_color(prob):
#     if prob <= 25:
#         return "blue"
#     elif prob <= 50:
#         return "yellow"
#     elif prob <= 75:
#         return "orange"
#     else:
#         return "red"


# # -----------------------------------------------------------
# # TAB 1 — SINGLE PREDICTION
# # -----------------------------------------------------------
# with tab1:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Single Person Prediction")

#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", 1, 120, 45)
#         sex_input = st.selectbox("Sex", ["Male", "Female"])
#         chest_pain_input = st.selectbox("Chest Pain Type",
#             ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
#         resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
#         cholesterol = st.number_input("Cholesterol", 0, 600, 200)

#     with col2:
#         fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
#         resting_ecg_input = st.selectbox("Resting ECG",
#             ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
#         max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
#         exercise_angina_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
#         st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

#     # Encode Inputs
#     sex = 1 if sex_input == "Male" else 0
#     chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
#     fast = 1 if fasting_bs == "Yes" else 0
#     ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
#     angina = 1 if exercise_angina_input == "Yes" else 0
#     slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

#     input_df = pd.DataFrame({
#         "Age":[age], "Sex":[sex], "ChestPainType":[chest],
#         "RestingBP":[resting_bp], "Cholesterol":[cholesterol],
#         "FastingBS":[fast], "RestingECG":[ecg], "MaxHR":[max_hr],
#         "ExerciseAngina":[angina], "ST_Slope":[slope]
#     })

#     if st.button(" Predict Now") and model is not None:
#         pred = model.predict(input_df)[0]
#         prob = float(model.predict_proba(input_df)[0][1] * 100)

#         if pred == 1:
#             st.markdown("<div class='result-positive'> Heart Disease Risk Detected</div>", unsafe_allow_html=True)
#         else:
#             st.markdown("<div class='result-negative'> No Heart Disease Detected</div>", unsafe_allow_html=True)

#         # --- GAUGE CHART WITH COLOR CONTROL ---
#         gauge_color = get_gauge_color(prob)

#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=round(prob, 2),
#             title={'text': "Risk Probability (%)"},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': gauge_color}
#             }
#         ))
#         st.plotly_chart(fig, use_container_width=True)

#         st.subheader("💡 Health Suggestions")
#         for tip in get_health_suggestions(input_df.iloc[0]):
#             st.write(f"- {tip}")

#     st.markdown('</div>', unsafe_allow_html=True)


# # -----------------------------------------------------------
# # TAB 2 — BULK PREDICT (CSV + EXCEL)
# # -----------------------------------------------------------
# with tab2:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Upload CSV or Excel File")

#     uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

#     # Category → numeric mappings
#     sex_map = {"Male": 1, "Female": 0}
#     chest_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
#     ecg_map = {"Normal": 0, "ST-T wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
#     angina_map = {"No": 0, "Yes": 1}
#     slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

#     if uploaded_file:
#         try:
#             # Read file
#             if uploaded_file.name.endswith(".csv"):
#                 df = pd.read_csv(uploaded_file)
#             else:
#                 df = pd.read_excel(uploaded_file, engine="openpyxl")

#             # Apply encoding on text columns
#             mapping_dict = {
#                 "Sex": sex_map,
#                 "ChestPainType": chest_map,
#                 "RestingECG": ecg_map,
#                 "ExerciseAngina": angina_map,
#                 "ST_Slope": slope_map
#             }

#             for col, mapping in mapping_dict.items():
#                 if col in df.columns and df[col].dtype == object:
#                     df[col] = df[col].map(mapping)

#             # Predict
#             preds = model.predict(df)
#             probs = model.predict_proba(df)[:, 1] * 100

#             df_out = df.copy()
#             df_out["Prediction"] = preds
#             df_out["Probability"] = np.round(probs, 2)
#             df_out["Suggestions"] = df_out.apply(lambda r: "; ".join(get_health_suggestions(r)), axis=1)

#             st.success("Predictions completed!")
#             st.dataframe(df_out)

#             st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"❌ Error: {e}")

#     st.markdown('</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------
# # TAB 3 — MODEL INSIGHTS (Blue Palette)
# # -----------------------------------------------------------
# with tab3:
#     st.markdown('<div class="container-card">', unsafe_allow_html=True)
#     st.subheader("Model Accuracy — Blue Palette 🎨")

#     accuracy = {
#         "Decision Tree": 80.97,
#         "Logistic Regression": 85.86,
#         "Random Forest": 84.23,
#         "SVM": 89.75
#     }

#     df_acc = pd.DataFrame(accuracy.items(), columns=["Model", "Accuracy"])
#     st.dataframe(df_acc)

#     blue_palette = ["#89CFF0", "#0A84FF", "#0074E8", "#003A92"]

#     fig = go.Figure(go.Bar(
#         x=df_acc["Model"],
#         y=df_acc["Accuracy"],
#         marker=dict(color=blue_palette)
#     ))

#     fig.update_layout(
#         title="Model Accuracy Comparison",
#         yaxis=dict(range=[0, 100])
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     st.markdown('</div>', unsafe_allow_html=True)










import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import base64

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Heart Health — iOS Sky UI",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="💙"
)

# -----------------------------------------------------------
# THEME COLORS
# -----------------------------------------------------------
BG_TOP = "#f6fbff"
BG_BOTTOM = "#e9f7ff"
CARD_BG = "rgba(255,255,255,0.85)"
ACCENT = "#0A84FF"
TEXT = "#0B2545"
SUBTEXT = "#54657A"
SHADOW = "rgba(10,132,255,0.08)"
INSIGHT_BLUE = "#003A92"

st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(160deg, #000000 0%, #0d1b2a 40%, #1b263b 80%);
  min-height: 100vh;
}}

html, body, [class*="css"] {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
  color: white;
}}

.container-card {{
    background: rgba(255,255,255,0.05);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 12px 50px rgba(0,0,0,0.7);
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}}

.result-positive {{
    background: rgba(255, 77, 77, 0.2);
    padding: 12px;
    border-radius: 12px;
    border-left: 6px solid #FF453A;
    font-weight: 600;
    color: #ffaaaa;
}}

.result-negative {{
    background: rgba(52, 199, 89, 0.2);
    padding: 12px;
    border-radius: 12px;
    border-left: 6px solid #34C759;
    font-weight: 600;
    color: #afffba;
}}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# DOWNLOAD CSV HELPER
# -----------------------------------------------------------
def get_csv_download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇ Download CSV</a>'


# -----------------------------------------------------------
# HEALTH SUGGESTION ENGINE
# -----------------------------------------------------------
def get_health_suggestions(row):
    tips = []

    if row["RestingBP"] > 130:
        tips.append("High BP — reduce salt and monitor weekly.")
    if row["Cholesterol"] > 240:
        tips.append("High cholesterol — avoid oily & fried foods.")
    if row["FastingBS"] == 1:
        tips.append("High blood sugar — reduce sugar and rice.")
    if row["MaxHR"] < 120:
        tips.append("Low Max HR — walk or do light cardio daily.")
    if row["ExerciseAngina"] == 1:
        tips.append("Exercise-induced angina — avoid heavy workouts.")
    if row["RestingECG"] != 0:
        tips.append("ECG abnormality detected — follow up with cardiologist.")
    if row["ST_Slope"] == 2:
        tips.append("Downsloping ST — seek medical attention soon.")
    if row["ChestPainType"] == 0:
        tips.append("Typical Angina — possible coronary blockage.")

    return tips[:5]


# -----------------------------------------------------------
# HEADER UI
# -----------------------------------------------------------
st.markdown('<div class="container-card">', unsafe_allow_html=True)
st.markdown("""
<h1 style="text-align:center;">💙 Heart Disease Prediction</h1>
<div style="text-align:center; color:#54657A;">AI-Powered Health Analysis</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📁 Bulk Predict", "📊 Model Insights"])


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
try:
    model = pickle.load(open("LogisticR.pkl", "rb"))
except:
    model = None
    st.error("❌ LogisticR.pkl not found in folder.")


# -----------------------------------------------------------
# GAUGE COLOR FUNCTION
# -----------------------------------------------------------
def get_gauge_color(prob):
    if prob <= 25:
        return "blue"
    elif prob <= 50:
        return "yellow"
    elif prob <= 75:
        return "orange"
    else:
        return "red"


# -----------------------------------------------------------
# TAB 1 — SINGLE PREDICTION
# -----------------------------------------------------------
with tab1:
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.subheader("Single Person Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex_input = st.selectbox("Sex", ["Male", "Female"])
        chest_pain_input = st.selectbox("Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        resting_bp = st.number_input("Resting Blood Pressure", 0, 300, 120)
        cholesterol = st.number_input("Cholesterol", 0, 600, 200)

    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dl?", ["No", "Yes"])
        resting_ecg_input = st.selectbox("Resting ECG",
            ["Normal", "ST-T wave Abnormality", "Left Ventricular Hypertrophy"])
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
        exercise_angina_input = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        st_slope_input = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # Encode Inputs
    sex = 1 if sex_input == "Male" else 0
    chest = ["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"].index(chest_pain_input)
    fast = 1 if fasting_bs == "Yes" else 0
    ecg = ["Normal","ST-T wave Abnormality","Left Ventricular Hypertrophy"].index(resting_ecg_input)
    angina = 1 if exercise_angina_input == "Yes" else 0
    slope = ["Upsloping","Flat","Downsloping"].index(st_slope_input)

    input_df = pd.DataFrame({
        "Age":[age], "Sex":[sex], "ChestPainType":[chest],
        "RestingBP":[resting_bp], "Cholesterol":[cholesterol],
        "FastingBS":[fast], "RestingECG":[ecg], "MaxHR":[max_hr],
        "ExerciseAngina":[angina], "ST_Slope":[slope]
    })

    if st.button(" Predict Now") and model is not None:
        pred = model.predict(input_df)[0]
        prob = float(model.predict_proba(input_df)[0][1] * 100)

        if pred == 1:
            st.markdown("<div class='result-positive'> Heart Disease Risk Detected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-negative'> No Heart Disease Detected</div>", unsafe_allow_html=True)

        gauge_color = get_gauge_color(prob)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob, 2),
            title={'text': "Risk Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💡 Health Suggestions")
        for tip in get_health_suggestions(input_df.iloc[0]):
            st.write(f"- {tip}")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------
# TAB 2 — BULK PREDICT (CSV + EXCEL)
# -----------------------------------------------------------
with tab2:
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.subheader("Upload CSV or Excel File")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

    # Category → numeric mappings
    sex_map = {"Male": 1, "Female": 0}
    chest_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
    ecg_map = {"Normal": 0, "ST-T wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
    angina_map = {"No": 0, "Yes": 1}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine="openpyxl")

            mapping_dict = {
                "Sex": sex_map,
                "ChestPainType": chest_map,
                "RestingECG": ecg_map,
                "ExerciseAngina": angina_map,
                "ST_Slope": slope_map
            }

            for col, mapping in mapping_dict.items():
                if col in df.columns and df[col].dtype == object:
                    df[col] = df[col].map(mapping)

            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1] * 100

            df_out = df.copy()
            df_out["Prediction"] = preds
            df_out["Probability"] = np.round(probs, 2)
            df_out["Suggestions"] = df_out.apply(lambda r: "; ".join(get_health_suggestions(r)), axis=1)

            st.success("Predictions completed!")
            st.dataframe(df_out)

            st.markdown(get_csv_download_link(df_out), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------
# TAB 3 — MODEL INSIGHTS (Blue Palette)
# -----------------------------------------------------------
with tab3:
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.subheader("Model Accuracy — Blue Palette 🎨")

    accuracy = {
        "Decision Tree": 80.97,
        "Logistic Regression": 85.86,
        "Random Forest": 84.23,
        "SVM": 89.75
    }

    df_acc = pd.DataFrame(accuracy.items(), columns=["Model", "Accuracy"])
    st.dataframe(df_acc)

    blue_palette = ["#89CFF0", "#0A84FF", "#0074E8", "#003A92"]

    fig = go.Figure(go.Bar(
        x=df_acc["Model"],
        y=df_acc["Accuracy"],
        marker=dict(color=blue_palette)
    ))

    fig.update_layout(
        title="Model Accuracy Comparison",
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
