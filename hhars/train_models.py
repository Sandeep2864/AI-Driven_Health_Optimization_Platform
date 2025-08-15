# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    patients = pd.read_csv(f"{DATA_DIR}/patients.csv", dtype=str)
    # If column names differ, adjust. Parse numeric columns:
    num_cols = ['Age','Height_cm','Weight_kg','BMI','Blood_Pressure_Systolic',
                'Blood_Pressure_Diastolic','Cholesterol_Level','Blood_Sugar_Level',
                'Daily_Steps','Exercise_Frequency','Sleep_Hours','Caloric_Intake',
                'Protein_Intake','Carbohydrate_Intake','Fat_Intake',
                'Recommended_Calories','Recommended_Protein','Recommended_Carbs','Recommended_Fats']
    for c in num_cols:
        if c in patients.columns:
            patients[c] = pd.to_numeric(patients[c], errors='coerce')
    # Fill missing numeric with medians
    patients.fillna(patients.median(numeric_only=True), inplace=True)
    patients.fillna('Unknown', inplace=True)
    return patients

def feature_engineer(df):
    # Example: derive BMI category and simple risk label
    df = df.copy()
    df['bmi_cat'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100], labels=['Underweight','Normal','Overweight','Obese'])
    # Simple risk label heuristic (for training a classifier) - only for demo
    def risk_row(r):
        # high risk if diabetes or heart disease or BMI>30 or BP high
        if r['Chronic_Disease'] in ['Diabetes','Heart Disease','Hypertension']:
            return 'High'
        if r['BMI'] >= 30:
            return 'High'
        if (r['Blood_Pressure_Systolic'] >= 140) or (r['Blood_Sugar_Level'] >= 180):
            return 'Medium'
        return 'Low'
    df['risk_label'] = df.apply(risk_row, axis=1)
    return df

def train():
    patients = load_data()
    patients = feature_engineer(patients)
    # target regression = Recommended_Calories, Recommended_Protein etc.
    target_cols = ['Recommended_Calories','Recommended_Protein','Recommended_Carbs','Recommended_Fats']
    # features
    features = ['Age','Gender','BMI','Daily_Steps','Exercise_Frequency','Sleep_Hours','Dietary_Habits','Chronic_Disease']
    X = patients[features]
    y_reg = patients[target_cols].ffill()
    y_cls = patients['risk_label']

    # Preprocessing
    numeric_features = ['Age','BMI','Daily_Steps','Exercise_Frequency','Sleep_Hours']
    categorical_features = ['Gender','Dietary_Habits','Chronic_Disease']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Regressor pipeline (multi-output)
    reg = Pipeline([
        ('prep', preprocessor),
        ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
    ])

    # We'll train separate regressors for calories and protein (simpler)
    X_train, X_test, y_train, y_test, y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=0.2, random_state=42
    )

    # Train 4 regressors (calories/protein/carbs/fats) using same pipeline
    models = {}
    for col in target_cols:
        r = Pipeline([
            ('prep', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))
        ])
        r.fit(X_train, y_train[col])
        models[col] = r
        print(f"Trained regressor for {col}")

    # Train classifier for risk_label
    cls = Pipeline([
        ('prep', preprocessor),
        ('model', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    cls.fit(X_train, y_cls_train)
    print("Trained risk classifier")

    joblib.dump(models, f"{MODEL_DIR}/regressors.pkl")
    joblib.dump(cls, f"{MODEL_DIR}/risk_model.pkl")
    print("Saved models to models/")

if __name__ == "__main__":
    train()
