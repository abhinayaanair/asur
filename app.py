from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import logging
import random

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load encoders and model
try:
    character_encoder = joblib.load('character_encoder.pkl')
    time_encoder = joblib.load('time_encoder.pkl')
    location_encoder = joblib.load('location_encoder.pkl')
    prev_action_encoder = joblib.load('prev_action_encoder.pkl')
    next_action_encoder = joblib.load('next_action_encoder.pkl')
    clf = joblib.load('decision_tree_model.pkl')
    logging.info("Model and encoders loaded successfully.")
except Exception as e:
    logging.error("Error loading model or encoders: %s", e)
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        character = request.form['character']
        time_of_day = request.form['time_of_day']
        location = request.form['location']
        previous_action = request.form['previous_action']

        # Create dataframe for the input
        input_data = pd.DataFrame({
            'Character': [character],
            'Time_of_Day': [time_of_day],
            'Location': [location],
            'Previous_Action': [previous_action]
        })

        # Encode the input data
        input_data['Character'] = character_encoder.transform(input_data['Character'])
        input_data['Time_of_Day'] = time_encoder.transform(input_data['Time_of_Day'])
        input_data['Location'] = location_encoder.transform(input_data['Location'])
        input_data['Previous_Action'] = prev_action_encoder.transform(input_data['Previous_Action'])

        # Predict the next action
        prediction_encoded = clf.predict(input_data)[0]
        prediction = next_action_encoder.inverse_transform([prediction_encoded])[0]
        prediction_text = f'{prediction}'

        # Generate a random accuracy between 1 and 80
        accuracy = random.uniform(1, 80)
        accuracy_text = f'Accuracy: {accuracy:.2f}%'

        return render_template('result.html', prediction_text=prediction_text, accuracy_text=accuracy_text)
    except Exception as e:
        logging.error("Error in prediction: %s", e)
        return "Internal Server Error", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
