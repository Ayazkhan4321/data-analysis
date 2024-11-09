from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load your model (assuming it's saved as 'student_performance_model.pkl')
model = joblib.load('student_performance_model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Student Performance Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Convert JSON to DataFrame for prediction
    df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(df)[0]

    # Return the result
    result = "Pass" if prediction == 1 else "Fail"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
