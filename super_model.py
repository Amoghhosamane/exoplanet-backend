import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os

# Define the expected features the model will use
MODEL_FEATURES = ['feature_1', 'feature_2', 'feature_3']
MODEL_FILENAME = 'model.joblib'

def create_and_save_model():
    """
    Creates a simple dummy LightGBM model and saves it to 'model.joblib'.
    """
    print("--- Starting model setup and saving ---")
    
    # 1. Generate Dummy Data (REPLACE THIS WITH YOUR REAL DATA LOADING AND PREPROCESSING)
    np.random.seed(42)
    data_size = 100
    features = {
        'feature_1': np.random.rand(data_size) * 10,
        'feature_2': np.random.rand(data_size) * 5,
        'feature_3': np.random.randint(0, 5, data_size)
    }
    df = pd.DataFrame(features)
    # Target is a simple function of features + some noise
    df['target'] = (df['feature_1'] + df['feature_2'] * df['feature_3'] + 
                    np.random.normal(0, 1, data_size) > 10).astype(int)
    
    X = df[MODEL_FEATURES]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train the LightGBM Model
    lgb_clf = lgb.LGBMClassifier(random_state=42)
    lgb_clf.fit(X_train, y_train)
    
    # 3. Save the trained model using joblib
    joblib.dump(lgb_clf, MODEL_FILENAME)
    
    print(f"Successfully trained and saved model to: {MODEL_FILENAME}")
    print("--- Setup complete. You MUST commit 'model.joblib' to Git now before deploying. ---")

if __name__ == "__main__":
    create_and_save_model()
