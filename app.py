from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
model = load_model('lstm_minute_model.h5')

# Route for predicting
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prices = np.array(data['prices']).reshape(-1, 1)  # Minute-level prices

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    # Create input for model
    look_back = 60
    X_input = scaled_prices[-look_back:]
    X_input = np.reshape(X_input, (1, look_back, 1))

    # Make prediction
    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction)

    return jsonify({'predictions': prediction.tolist()[0]})

if __name__ == '__main__':
    app.run(debug=True)
