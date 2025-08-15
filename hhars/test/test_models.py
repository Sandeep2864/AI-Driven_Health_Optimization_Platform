import os
import pytest
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# Path configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def test_regression_models():
    """Test regression models using patient data with realistic expectations"""
    # Load data
    patients_path = os.path.join(DATA_DIR, "patients.csv")
    if not os.path.exists(patients_path):
        pytest.skip(f"Patient data file not found at {patients_path}")

    data = pd.read_csv(patients_path)

    # Load models
    regressors_path = os.path.join(MODEL_DIR, "regressors.pkl")
    if not os.path.exists(regressors_path):
        pytest.skip(f"Regressors model not found at {regressors_path}")

    regressors = load(regressors_path)

    # Prepare features
    feature_cols = ['Age', 'Gender', 'BMI', 'Daily_Steps',
                    'Exercise_Frequency', 'Sleep_Hours',
                    'Dietary_Habits', 'Chronic_Disease']
    X = data[feature_cols]

    # Test each regressor
    regression_metrics = {}
    for target, model in regressors.items():
        y_true = data[target]
        y_pred = model.predict(X)

        regression_metrics[target] = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'CV_MAE': cross_val_score(
                model, X, y_true, cv=5,
                scoring='neg_mean_absolute_error'
            ).mean() * -1,
            'CV_R2': cross_val_score(model, X, y_true, cv=5, scoring='r2').mean()
        }

    # Print metrics
    print("\nRegression Model Metrics:")
    for target, metrics in regression_metrics.items():
        print(f"\n{target}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # More realistic assertions
    for target in regressors:
        assert regression_metrics[target]['R2'] > 0.1, f"{target} model R2 too low"
        assert regression_metrics[target]['MAE'] < 1000, f"{target} model MAE too high"


def test_classifier_model():
    """Test risk classifier using patient data"""
    # Load data
    patients_path = os.path.join(DATA_DIR, "patients.csv")
    if not os.path.exists(patients_path):
        pytest.skip(f"Patient data file not found at {patients_path}")

    data = pd.read_csv(patients_path)

    # Skip test if risk_label column doesn't exist
    if 'risk_label' not in data.columns:
        pytest.skip("'risk_label' column not found in test data")

    # Load model
    classifier_path = os.path.join(MODEL_DIR, "risk_model.pkl")
    if not os.path.exists(classifier_path):
        pytest.skip(f"Classifier model not found at {classifier_path}")

    classifier = load(classifier_path)

    # Prepare features and target
    feature_cols = ['Age', 'Gender', 'BMI', 'Daily_Steps',
                    'Exercise_Frequency', 'Sleep_Hours',
                    'Dietary_Habits', 'Chronic_Disease']
    X = data[feature_cols]
    y_true = data['risk_label']

    # Calculate accuracy
    y_pred = classifier.predict(X)
    accuracy = (y_pred == y_true).mean()

    print(f"\nClassifier Accuracy: {accuracy:.4f}")

    # Assert reasonable accuracy
    assert accuracy > 0.7, "Classifier accuracy too low"