import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# --- 1. Create Dummy Data (Mimicking the real IPL features) ---
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals', 'Kings XI Punjab']
cities = ['Mumbai', 'Delhi', 'Kolkata']

# Create a small, balanced dummy dataset (500 rows)
data = {
    'batting_team': np.random.choice(teams, 500),
    'bowling_team': np.random.choice(teams, 500),
    'city': np.random.choice(cities, 500),
    'runs_left': np.random.randint(5, 150, 500),
    'balls_left': np.random.randint(10, 100, 500),
    'wickets_left': np.random.randint(3, 10, 500),
    'target': np.random.randint(120, 200, 500),
    'crr': np.random.uniform(5.0, 10.0, 500),
    'rrr': np.random.uniform(5.0, 15.0, 500),
    'result': np.random.randint(0, 2, 500) 
}

df = pd.DataFrame(data)
df = df[df['batting_team'] != df['bowling_team']].reset_index(drop=True)

X = df.drop(columns=['result'])
y = df['result']

# --- 2. Create and Save the Dummy Pipeline ---
trf = ColumnTransformer([
    ('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
     ['batting_team', 'bowling_team', 'city'])
], remainder='passthrough')

pipe = Pipeline(steps=[
    ('step1', trf),
    ('step2', LogisticRegression(solver='liblinear'))
])

# Train the Dummy Model (Replace this step with your actual training code for real accuracy)
pipe.fit(X, y)

# Save the Pipeline
pickle.dump(pipe, open('pipe.pkl', 'wb'))

print("Dummy model 'pipe.pkl' created successfully. You can now run app.py.")