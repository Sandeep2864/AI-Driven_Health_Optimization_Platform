import sys
from pathlib import Path
import pytest
import json

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture
def client():
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    """Test /predict endpoint"""
    user_data = {
        'age': 30,
        'gender': 'Male',
        'bmi': 24,
        'daily_steps': 5000,
        'exercise_freq': 3,
        'sleep_hours': 7,
        'diet': 'Vegetarian',
        'chronic': 'None',
        'allergies': 'Dairy'
    }
    response = client.post('/predict', json=user_data)
    assert response.status_code == 200
    pred = json.loads(response.data)
    assert 'recommended' in pred
    assert 'calories' in pred['recommended']

def test_recommend_endpoint(client):
    """Test /recommend endpoint"""
    rec_data = {
        'age': 30,
        'gender': 'Male',
        'bmi': 24,
        'allergies': 'Dairy',
        'recommended': {
            'calories': 2000,
            'protein': 100,
            'carbs': 250,
            'fats': 70
        }
    }
    response = client.post('/recommend', json=rec_data)
    assert response.status_code == 200
    week_plan = json.loads(response.data)
    assert len(week_plan['week_plan']) == 7
    for day in week_plan['week_plan']:
        assert len(day) > 0  # At least one meal per day