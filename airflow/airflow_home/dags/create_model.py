import pandas as pd  
import mlflow
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data():
    pa = "C:/Users/harsh/fin/data/states.csv"
    df = pd.read_csv(pa)
    
    # Drop nulls and unnecessary columns
    df = df.dropna()
    df = df.drop(columns=['Uninsured Rate Change (2010-2015)'])
    
    # Clean percentage and dollar values
    def clean_percentage(value):
        if isinstance(value, str):
            if "%" in value:
                return value.replace('%', '')
            elif "$" in value:
                return value.replace('$', '').replace(',', '')
        return value
    
    # Apply cleaning to all columns
    df = df.applymap(clean_percentage)
    
    # Convert object columns to float
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Select features and target
    x = df.iloc[:,[3,4,5,6,7,9,10,11,12]].values
    y = LabelEncoder().fit_transform(df.iloc[:,8])
    
    return x, y


def scale_data(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Save the data using os.path.join for cross-platform compatibility
    base_path = os.path.join('C:', os.sep, 'Users', 'harsh', 'fin')
    pd.DataFrame(x_train).to_csv(os.path.join(base_path, 'x_train.csv'))
    pd.DataFrame(y_train).to_csv(os.path.join(base_path, 'y_train.csv'))
    
    return x_train, x_test, y_train, y_test


def train_model():
    # Use os.path.join for cross-platform compatibility
    base_path = os.path.join('C:', os.sep, 'Users', 'harsh', 'fin')
    x_path = os.path.join(base_path, 'x_train.csv')
    y_path = os.path.join(base_path, 'y_train.csv')
        
    clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,
                        activation = 'relu',solver='adam',random_state=1)
    
    x = pd.read_csv(x_path).iloc[:, 1:]
    y = pd.read_csv(y_path).iloc[:, 1]

    clf.fit(x,y)

    with mlflow.start_run():
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.log_param('hidden_layer_sizes', (150, 100, 50))
        mlflow.log_param('max_iter', 300)
        mlflow.log_param('activation', 'relu')
        mlflow.log_param('solver', 'adam')
        mlflow.sklearn.log_model(clf, 'MLP1')
    
    print("Task Complete!")
    return True

if __name__ == "__main__":
    x, y = preprocess_data()
    x_scaled = scale_data(x)
    x_train, x_test, y_train, y_test = split_data(x_scaled, y)
    train_model()







