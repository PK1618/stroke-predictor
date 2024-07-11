from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import logging

app = Flask(__name__)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

logging.basicConfig(level=logging.INFO)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    logging.info(f"Form data: {data}")

    user_input = {
        'sex': 1 if data['gender'] == 'Male' else 0,
        'age': int(data['age']),
        'hypertension': 1 if data['ht'] == 'Yes' else 0,
        'heart_disease': 1 if data['hd'] == 'Yes' else 0,
        'ever_married': 1 if data['marriage'] == 'Yes' else 0,
        'work_type': int(data['WorkType']),
        'Residence_type': 1 if data['Residence'] == 'Urban' else 0,
        'avg_glucose_level': float(data['Glucose']),
        'bmi': float(data['BMI']),
        'smoking_status': 1 if data['smoker'] == 'Yes' else 0
    }

    logging.info(f"User input data: {user_input}")
    
    user_input_df = pd.DataFrame([user_input])
    user_scaler = scaler.transform(user_input_df)

    logging.info(f"Scaled user input data: {user_scaler}")

    prediction_probabilities = model.predict(user_scaler)
    predictions = (prediction_probabilities > 0.5).astype(int)

    logging.info(f"Prediction: {predictions}")
    logging.info(f"Prediction probabilities: {prediction_probabilities}")

    # Prepare the result
    prediction_result = predictions[0][0]
    prediction_probability = prediction_probabilities[0][0]

    result = {
        'prediction': 'Positive' if prediction_result == 1 else 'Negative',
        'probability': prediction_probability
    }

    logging.info(f"Result: {result}")


    # Render the result in the template
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)