from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the model and scaler
model = load_model('lstm_5months_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess the input data as per your model's requirement
    input_data = np.array(data['input']).reshape((1, -1, 1))  # Adjust based on your preprocessing logic
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
