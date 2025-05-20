# src/train.py
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn
import os
import yaml
import pickle
import json

#load parameters of the models
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

## -- MLFLOW SETUP -- ##
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('mlp_model')

#start a mlflow run
with mlflow.start_run():
    seed = params['base']['seed']
    model_params = params['train']

    #log the parameters here
    mlflow.log_param('seed', seed)
    mlflow.log_params(model_params)

    #load data
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')

    # Initialize LabelEncoder and StandardScaler
    le = LabelEncoder()
    scaler = StandardScaler()

    
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

    X_train = train_df[feature_cols].values
    y_train = le.fit_transform(train_df[target_col])

    X_test = val_df[feature_cols].values
    y_test = le.transform(val_df[target_col])


    X_train = train_df[feature_cols].values
    y_train = le.fit_transform(train_df[target_col])

    X_test = val_df[feature_cols].values
    y_test = le.transform(val_df[target_col])


    # Scale the features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Train Model ---
    if model_params['model_type'] == 'MLPClassifier':
        model = MLPClassifier(
            hidden_layer_sizes=tuple(model_params['hidden_layer_sizes']),
            max_iter=model_params['max_iter'],
            activation=model_params['activation'],
            solver=model_params['solver'],
            random_state=seed
        )
    else:
        raise ValueError(f"Unsupported model type: {model_params['model_type']}")
    
    #train the model
    model.fit(X_train, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")

       # --- Save and Log Model ---
    os.makedirs('models', exist_ok=True)
    model_output_path = 'models/model.pkl'
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save the model
    mlflow.sklearn.log_model(model, "mlp_model")
    print(f"Model saved to: {model_output_path}")
    print(f"Model logged to MLflow run: {mlflow.active_run().info.run_id}")
   
     # --- Save Metrics File (for DVC tracking) ---
    metrics_output_path = 'metrics.json'
    metrics_data = {'accuracy': accuracy}
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_output_path}")

print("MLflow Run Completed.")