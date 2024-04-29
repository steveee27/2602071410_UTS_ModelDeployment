import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encoders
binary_encode = joblib.load('Binary_Encode.pkl')
scaler = joblib.load('StandardScaler.pkl')
model = joblib.load('xgb_classifier_model.pkl')

def main():
    st.title('Churn Prediction')

    # User input components
    creditscore = st.number_input("Credit Score", min_value=0)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=0)
    tenure = st.number_input("Tenure", min_value=0)
    balance = st.number_input("Balance", min_value=0.0)
    numofproducts = st.selectbox("Number of Products", [1, 2, 3, 4])
    hascrcard = st.selectbox("Has Credit Card", ['Yes', 'No'])
    isactivemember = st.selectbox("Is Active Member", ['Yes', 'No'])
    estimatedsalary = st.number_input("Estimated Salary", min_value=0.0)

    # Convert categorical inputs to encoded form
    gender_encoded = binary_encode['Gender'][gender]
    hascrcard_encoded = binary_encode['HasCrCard'][hascrcard]
    isactivemember_encoded = binary_encode['IsActiveMember'][isactivemember]

    # Create input DataFrame
    input_data = {
        'CreditScore': creditscore,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': numofproducts,
        'HasCrCard': hascrcard_encoded,
        'IsActiveMember': isactivemember_encoded,
        'EstimatedSalary': estimatedsalary
    }
    input_df = pd.DataFrame([input_data])

    # Scale numerical inputs
    scaled_inputs = scaler.transform(input_df)

    if st.button('Predict'):
        prediction = make_prediction(scaled_inputs)
        st.success(f'The prediction is: {prediction}')

def make_prediction(features):
    # Use the loaded model to make predictions
    prediction = model.predict(features)
    return prediction[0]

if __name__ == '__main__':
    main()
