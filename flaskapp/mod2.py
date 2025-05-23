import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from benfordslaw import benfordslaw
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

url = "https://raw.githubusercontent.com/HarshiniAiyyer/Financial-Forensics/refs/heads/main/states.csv"

df = pd.read_csv(url)

df.isnull().sum()

df = df.dropna()
df = df.drop(columns = ['Uninsured Rate Change (2010-2015)'])

# Remove the percentages and dollar signs
def clean_percentage(value):
    if isinstance(value, str):
        if "%" in value:
            return value.replace('%', '')
        elif "$" in value:
            return value.replace('$', '').replace(',', '')
    return value

# Apply the cleaning function to all columns
df = df.map(clean_percentage)

# Loop through columns (excluding the first column) and convert 'object' columns to float
for col in df.columns[1:]:  # Exclude the first column by starting from index 1
    if df[col].dtype == 'object':  # Check if the column has 'object' type
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric (float), set errors to NaN if conversion fails


"""### ML algorithms Pipeline

#### Data Setup
"""

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

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

# Scale the features
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))