from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        input_data = {feature: float(request.form[feature]) for feature in feature_names}
        
        # Create DataFrame and scale input
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]  # Ensure correct order
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        result = 'Fraud' if prediction == 1 else 'Not Fraud'
        
        return render_template('index.html',
                               prediction_text=f"Prediction: {result}",
                               feature_names=feature_names)
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}",
                               feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
