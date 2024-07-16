# from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
# import logging
import streamlit as st

# app = Flask(__name__)
def load_model_and_scaler():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# logging.basicConfig(level=logging.INFO)


# @app.route('/')
def welcome():
    return "Welcome"

def predict(sex, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status):
    
    sex_map = {'Male': 1, 'Female': 0}
    hypertension_map = {'Yes': 1, 'No': 0}
    heart_disease_map = {'Yes': 1, 'No': 0}
    ever_married_map = {'Yes': 1, 'No': 0}
    work_type_map = {'No Work': 1, 'Government Job': 2, 'Self-Employed': 3, 'Private': 4}
    residence_type_map = {'Urban': 1, 'Rural': 0}
    smoking_status_map = {'Yes': 1, 'No': 0}
    
    sex = sex_map[sex]
    hypertension = hypertension_map[hypertension]
    heart_disease = heart_disease_map[heart_disease]
    ever_married = ever_married_map[ever_married]
    work_type = work_type_map[work_type]
    Residence_type = residence_type_map[Residence_type]
    smoking_status = smoking_status_map[smoking_status]

    user_df = pd.DataFrame([[sex, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]],
                           columns=['sex', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    user_scaler = scaler.transform(user_df)
    prediction = model.predict(user_scaler)
    print(prediction[0])
    return prediction[0]
    


# @app.route('/predict', methods=['POST'])
def main():
    st.title('Heart Stroke Predictor')
    st.markdown('---')
    sex = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Pick your age", 1, 110, 30)
    hypertension = st.radio("Do you have hypertension?", ['Yes', 'No'], horizontal=True, key = 'ht')
    heart_disease = st.radio("Do you have Heart disease before or now?", ['Yes', 'No'], horizontal=True, key='hd')
    ever_married = st.radio("Are you married?", ['Yes', 'No'], horizontal=True, key='marriage')
    work_type = st.radio("What is your work type?", ['No Work', 'Government Job', 'Self-Employed', 'Private'], horizontal=True, key='wt')
    Residence_type = st.selectbox("Select where you live", ['Urban', 'Rural'])
    avg_glucose_level = st.number_input("Enter your average glucose level")
    bmi = st.number_input("Enter your body mass index value")
    smoking_status = st.radio("Do you smoke?", ['Yes', 'No'], horizontal=True, key='smoke')

    result = ""

    if st.button('Predict'):
        result = predict(sex, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status)
        if result == 1:
            st.warning("POSITIVE!!! You might likely to get stroke. Control your food and maintain healthy life style.", icon="⚠️")
        else:
            st.success("NEGATIVE, You might not get heart stroke, maintain good lifestyle to avoid any in future.", icon="✅")
    return 
    

if __name__ == "__main__":
    main()




# data = request.form

#     logging.info(f"Form data: {data}")

#     user_input = {
#         'sex': 1 if data['gender'] == 'Male' else 0,
#         'age': int(data['age']),
#         'hypertension': 1 if data['ht'] == 'Yes' else 0,
#         'heart_disease': 1 if data['hd'] == 'Yes' else 0,
#         'ever_married': 1 if data['marriage'] == 'Yes' else 0,
#         'work_type': int(data['WorkType']),
#         'Residence_type': 1 if data['Residence'] == 'Urban' else 0,
#         'avg_glucose_level': float(data['Glucose']),
#         'bmi': float(data['BMI']),
#         'smoking_status': 1 if data['smoker'] == 'Yes' else 0
#     }

#     logging.info(f"User input data: {user_input}")
    
#     user_input_df = pd.DataFrame([user_input])
#     user_scaler = scaler.transform(user_input_df)

#     logging.info(f"Scaled user input data: {user_scaler}")

#     predictions = model.predict(user_scaler)

#     logging.info(f"Prediction: {predictions}")
# #     logging.info(f"Prediction probabilities: {prediction_probabilities}")

#     # Prepare the result
#     prediction_result = predictions[0][0]
# #     prediction_probability = prediction_probabilities[0][0]

#     result = {
#         'prediction': 'Positive' if prediction_result == 1 else 'Negative',
#         # 'probability': prediction_probability
#     }

#     logging.info(f"Result: {result}")


#     # Render the result in the template
#     return render_template('result.html', result=result)