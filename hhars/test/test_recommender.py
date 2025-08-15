import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from recommender import load_meals, filter_meals, score_meals, plan_week

@pytest.fixture
def sample_meals():
    """Mock meal data for testing"""
    return pd.DataFrame({
        'Food Name': ['Pasta', 'Salad', 'Chicken Curry', 'Fish Fry'],
        'Total Calories': [500, 200, 600, 450],
        'Total Protein': [20, 10, 40, 30],
        'Total Carbs': [80, 30, 50, 20],
        'Total Fats': [10, 5, 25, 30],
        'Allergic Ingredients': ['None', 'Dairy', 'None', 'Seafood'],
        'Type': ['Dinner', 'Lunch', 'Dinner', 'Dinner'],
        'State': ['Italian', 'American', 'Indian', 'Indian'],
        'Total Sodium': [300, 200, 400, 350],
        'Total Sugar': [10, 5, 15, 8]
    })

def test_load_meals():
    """Test loading meals from CSV"""
    meals = load_meals()
    assert not meals.empty, "Meals data should not be empty"
    assert 'Food Name' in meals.columns, "Missing required column"

def test_filter_meals(sample_meals):
    # Test allergy filtering
    filtered = filter_meals(sample_meals, allergies=['Dairy'])
    assert 'Salad' not in filtered['Food Name'].values

    # Test cuisine preference
    filtered = filter_meals(sample_meals, cuisine_pref='Indian')
    assert set(filtered['Food Name']) == {'Chicken Curry', 'Fish Fry'}

    # Test max sodium
    filtered = filter_meals(sample_meals, max_sodium=350)
    assert all(filtered['Total Sodium'] <= 350)

def test_score_meals(sample_meals):
    target = {'calories': 500, 'protein': 30, 'carbs': 60, 'fats': 20}
    scores = score_meals(sample_meals, target)
    assert len(scores) == len(sample_meals)
    assert scores.argmax() == 0  # Pasta should be closest to target

def test_plan_week(sample_meals):
    target = {'calories': 2000, 'protein': 100, 'carbs': 250, 'fats': 70}
    week_plan = plan_week(sample_meals, target, meals_per_day=2)
    assert len(week_plan) == 7  # 7 days
    assert len(week_plan[0]) == 2  # 2 meals per day
    # Check no allergic meals are included
    assert 'Salad' not in [m['Food Name'] for day in week_plan for m in day]