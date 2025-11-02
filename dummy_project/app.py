from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS
import numpy as np

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app) 

# Load the trained pipeline
try:
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    print("Model pipe.pkl loaded successfully.")
except FileNotFoundError:
    print("\nERROR: pipe.pkl not found! Did you run create_dummy_model.py?\n")
    pipe = None 

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if pipe is None:
        return jsonify({'error': 'Model file (pipe.pkl) is missing.'}), 500

    data = request.get_json()

    try:
        # Extract features
        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data['city']
        target = int(data['target'])
        score = int(data['score'])
        overs = float(data['overs'])
        wickets = int(data['wickets'])
    except Exception as e:
        return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

    # --- Feature Engineering (Calculations) ---
    runs_left = target - score
    
    # Calculate total balls bowled (e.g., 10.5 overs = 65 balls)
    total_balls_bowled = int(overs) * 6 + round(overs * 10 % 10) 
    balls_left = 120 - total_balls_bowled
    wickets_left = 10 - wickets
    
    # Calculate CRR and RRR
    crr = (score * 6) / total_balls_bowled if total_balls_bowled > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else np.inf
    
    # Create DataFrame for prediction
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Make prediction using the pipeline
    result = pipe.predict_proba(input_df)
    win_prob = result[0][1] # Probability of batting team winning
    loss_prob = result[0][0] # Probability of bowling team winning

    # Return the result as JSON
    return jsonify({
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'win_probability': win_prob * 100,
        'loss_probability': loss_prob * 100
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)