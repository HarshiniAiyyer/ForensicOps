
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

# Create processed data directory
os.makedirs('data/processed', exist_ok=True)

# Load parameters
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

split_ratio = params['prepare']['split']
seed = params['base']['seed']

# Load raw data
rawpath = 'data/raw/states.csv'
df = pd.read_csv(rawpath)

# --- Clean the data ---

#drop the rows with missing values
df = df.dropna()
#drop the column with the stupid minus symbol
df = df.drop(columns = ['Uninsured Rate Change (2010-2015)'])

# Example: Remove unwanted symbols and convert to numeric where needed
df.replace({'\$': '', '%': '', '\?': ''}, regex=True, inplace=True)

# Convert numeric columns properly
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Impute missing values if needed
df.fillna(df.mean(numeric_only=True), inplace=True)

# --- Split the cleaned data ---
train_df, val_df = train_test_split(df, test_size=split_ratio, random_state=seed)

# Save processed data
trainoutpath = 'data/processed/train.csv'
valoutpath = 'data/processed/val.csv'
train_df.to_csv(trainoutpath, index=False)
val_df.to_csv(valoutpath, index=False)

print(f"Processed data saved:")
print(f"- Train set: {trainoutpath}")
print(f"- Validation set: {valoutpath}")
