import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/Gradient_Boosting_Classifier_no_tuning_original_model.pkl')
model = joblib.load(MODEL_PATH)
# SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
# scaler = joblib.load(SCALER_PATH)

st.title('Diabetes Prediction App')
st.write('Enter patient data to predict diabetes risk:')

col1, col2 = st.columns(2)
with col1:
    st.markdown('#### Demographics & Vitals')
    genhlth = st.slider('General Health (1=Excellent, 5=Poor)', 1, 5, 3)
    highbp = st.selectbox('High Blood Pressure (0=No, 1=Yes)', [0, 1])
    age = st.slider('Age Category (1=18-24, ..., 13=80+)', 1, 13, 5)
    bmi = st.number_input('BMI', min_value=10.0, max_value=70.0, value=25.0)
    highchol = st.selectbox('High Cholesterol (0=No, 1=Yes)', [0, 1])
with col2:
    st.markdown('#### Lifestyle & Other Factors')
    sex = st.selectbox('Sex (0=Female, 1=Male)', [0, 1])
    income = st.slider('Income Scale (1=<$10k, 8=$75k+)', 1, 8, 4)
    hvyalcoholconsump = st.selectbox('Heavy Alcohol Consumption (0=No, 1=Yes)', [0, 1])
    cholcheck = st.selectbox('Cholesterol Check in 5 Years (0=No, 1=Yes)', [0, 1])
    physhlth = st.slider('Days Physical Health Not Good (0-30)', 0, 30, 0)
    heartdiseaseorattack = st.selectbox('Heart Disease or Attack (0=No, 1=Yes)', [0, 1])

# Place to store prediction result in session state
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
    st.session_state['prediction_proba'] = None

predict_clicked = st.button('Predict', use_container_width=True)

# Use session state to persist the last prediction and probabilities
if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None
    st.session_state['last_proba'] = None

if predict_clicked:
    feature_names = [
        'GenHlth','HighBP','Age','BMI','HighChol','Sex','Income',
        'HvyAlcoholConsump','CholCheck','PhysHlth','HeartDiseaseorAttack'
    ]
    input_df = pd.DataFrame([[genhlth, highbp, age, bmi, highchol, sex, income, hvyalcoholconsump, cholcheck, physhlth, heartdiseaseorattack]], columns=feature_names)
    # input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)
    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    st.session_state['last_prediction'] = result
    st.session_state['last_proba'] = proba

# Always display the last prediction if it exists
if st.session_state['last_prediction'] is not None:
    st.success(f'Prediction: {st.session_state["last_prediction"]}')
    proba = st.session_state['last_proba']
    st.info(f"Probability Not Diabetic: {proba[0][0]*100:.2f}%")
    st.info(f"Probability Diabetic: {proba[0][1]*100:.2f}%")