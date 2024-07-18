from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and encoders
clf = joblib.load('decision_tree_model.pkl')
le_character = joblib.load('character_encoder.pkl')
le_time = joblib.load('time_encoder.pkl')
le_location = joblib.load('location_encoder.pkl')
le_prev_action = joblib.load('prev_action_encoder.pkl')
le_next_action = joblib.load('next_action_encoder.pkl')

# For demonstration purposes, assuming an overall accuracy
model_accuracy = 0.85  # Example accuracy, replace with your model's accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    character = request.form['character']
    time_of_day = request.form['time_of_day']
    location = request.form['location']
    previous_action = request.form['previous_action']

    # Example dictionary
    example = {
        'Character': character,
        'Time_of_Day': time_of_day,
        'Location': location,
        'Previous_Action': previous_action
    }
    
    # Convert to DataFrame
    example_df = pd.DataFrame([example])

    # Encode the example using the same encoders
    example_df['Character'] = le_character.transform(example_df['Character'])
    example_df['Time_of_Day'] = le_time.transform(example_df['Time_of_Day'])
    example_df['Location'] = le_location.transform(example_df['Location'])
    example_df['Previous_Action'] = le_prev_action.transform(example_df['Previous_Action'])

    try:
        # Predict the next action
        predicted_next_action = clf.predict(example_df)
        predicted_next_action = le_next_action.inverse_transform(predicted_next_action)
        prediction_text = f'Predicted Next Action: {predicted_next_action[0]}'
        accuracy_info = f'Model Accuracy: {model_accuracy * 100:.2f}%'
    except Exception as e:
        prediction_text = f'Error predicting next action: {str(e)}'
        accuracy_info = ''

    return render_template('result.html', prediction_text=prediction_text, accuracy_info=accuracy_info)

if __name__ == '__main__':
    app.run(debug=True)
