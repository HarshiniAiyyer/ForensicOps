from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import os

app = Flask(__name__)

# Load and prepare the data
df = pd.read_csv('D:/yea/data/raw/states.csv')

# Clean the data
def clean_percentage(value):
    if isinstance(value, str):
        if "%" in value:
            return value.replace('%', '')
        elif "$" in value:
            return value.replace('$', '').replace(',', '')
    return value

# Apply cleaning to each column
for column in df.columns:
    df[column] = df[column].apply(clean_percentage)

# Convert numeric columns
for col in df.columns[1:]:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare features and target
feature_cols = [
    'Employer Health Insurance Coverage (2015)',
    'Marketplace Health Insurance Coverage (2016)',
    'Marketplace Tax Credits (2016)',
    'Average Monthly Tax Credit (2016)',
    'Medicaid Enrollment (2013)',
    'Medicaid Enrollment (2016)',
    'Medicaid Enrollment Change (2013-2016)',
    'Medicare Enrollment (2016)'
]

target_col = 'State Medicaid Expansion (2016)'

X = df[feature_cols].values
y = df[target_col].values

# Initialize and fit the scaler and encoder
scaler = StandardScaler()
le = LabelEncoder()

X_scaled = scaler.fit_transform(X)
y_encoded = le.fit_transform(y)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300, activation='relu', solver='adam', random_state=1)
model.fit(X_scaled, y_encoded)

@app.route('/')
def home():
    # Create display names mapping
    display_names = {
        'Employer Health Insurance Coverage (2015)': 'Employer Health Insurance Coverage in 2015',
        'Marketplace Health Insurance Coverage (2016)': 'Marketplace Health Insurance Coverage in 2016',
        'Marketplace Tax Credits (2016)': 'Marketplace Tax Credits in 2016',
        'Average Monthly Tax Credit (2016)': 'Average Monthly Tax Credit in 2016',
        'Medicaid Enrollment (2013)': 'Medicaid Enrollment in 2013',
        'Medicaid Enrollment (2016)': 'Medicaid Enrollment in 2016',
        'Medicaid Enrollment Change (2013-2016)': 'Medicaid Enrollment from 2013 to 2016',
        'Medicare Enrollment (2016)': 'Medicare Enrollment in 2016'
    }
    return render_template('index.html', features=feature_cols, display_names=display_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        features = []
        for feature in feature_cols:
            value = float(request.form[feature])
            features.append(value)
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_text = le.inverse_transform(prediction)[0]
        
        # Convert prediction to descriptive message
        if prediction_text == 0:
            result_message = "State Medicaid expansion has not been implemented."
        else:
            result_message = "State Medicaid expansion has been implemented."
        
        return jsonify({'prediction': result_message})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use 5000 for development, can be overridden by PORT env var
    app.run(debug=True, port=port) 