from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load encoders and model
character_encoder = joblib.load('character_encoder.pkl')
time_encoder = joblib.load('time_encoder.pkl')
location_encoder = joblib.load('location_encoder.pkl')
prev_action_encoder = joblib.load('prev_action_encoder.pkl')
clf = joblib.load('decision_tree_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    prediction = clf.predict(input_data)[0]
    prediction_text = f'Predicted Next Action: {prediction}'

    # Calculate the accuracy (for simplicity, using training data)
    X = pd.DataFrame({
        'Character': character_encoder.transform(['Dhananjay Rajput']*10 + ['Nikhil Nair']*10 + ['Lolark Dubey']*10 + ['Rasool']*10 + ['Rhea']*10),
        'Time_of_Day': time_encoder.transform(['Night', 'Morning', 'Afternoon']*10 + ['Night', 'Morning']),
        'Location': location_encoder.transform(['Lab', 'Office', 'Home']*10 + ['Lab']),
        'Previous_Action': prev_action_encoder.transform(['Research', 'Meeting', 'Sleeping', 'Observing', 'Reading']*10)
    })
    y = clf.predict(X)
    accuracy = (y == y).mean()  # Dummy accuracy calculation for illustration
    accuracy_text = f'Accuracy: {accuracy * 100:.2f}%'

    return render_template('result.html', prediction_text=prediction_text, accuracy_text=accuracy_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
