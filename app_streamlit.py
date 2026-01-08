# app_streamlit.py - Streamlit app to serve the Paisabazar loan eligibility model
import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = 'models/pipeline.joblib'

st.set_page_config(page_title='Paisabazar - Loan Eligibility Predictor', layout='centered')

st.title('Paisabazar — Loan Eligibility Predictor')

if not os.path.exists(MODEL_PATH):
    st.warning('Model not found. Please run train.py first to create models/pipeline.joblib')
else:
    model = joblib.load(MODEL_PATH)

    st.header('Applicant information')

    gender = st.selectbox('Gender', ['Male','Female'])
    married = st.selectbox('Married', ['Yes','No'])
    dependents = st.selectbox('Dependents', ['0','1','2','3+'])
    education = st.selectbox('Education', ['Graduate','Not Graduate'])
    self_emp = st.selectbox('Self Employed', ['Yes','No'])
    applicant_income = st.number_input('Applicant Income', min_value=0, value=5000)
    coapplicant_income = st.number_input('Coapplicant Income', min_value=0, value=0)
    loan_amount = st.number_input('Loan Amount (thousands)', min_value=1, value=140)
    loan_term = st.selectbox('Loan Term (months)', [360,180,120,240,300])
    credit_history = st.selectbox('Credit History', [1.0,0.0])
    prop_area = st.selectbox('Property Area', ['Urban','Semiurban','Rural'])

    if st.button('Predict Eligibility'):
        input_df = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_emp,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': prop_area
        }])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.success(f'Predicted: {"Eligible (Y)" if pred==1 else "Not Eligible (N)"} — Probability: {proba:.2f}')
