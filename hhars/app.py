from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json
import geocoder
import sys

# --- PATHING CONFIGURATION ---
# Simplified to help Flask find your 'static' folder easily
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Import logic from recommender.py
try:
    from recommender import load_meals, filter_meals, plan_week
except ImportError as e:
    print(f"❌ Error: {e}")

# This configuration is what makes the CSS work
app = Flask(__name__, 
            template_folder="static", 
            static_folder="static")
CORS(app)

# --- MODEL LOADING ---
REGRESSORS = None
RISK_MODEL = None

def load_models():
    global REGRESSORS, RISK_MODEL
    reg_path = os.path.join(MODEL_DIR, "regressors.pkl")
    risk_path = os.path.join(MODEL_DIR, "risk_model.pkl")

    if os.path.exists(reg_path):
        try:
            REGRESSORS = joblib.load(reg_path)
        except Exception as e:
            print(f"Error: {e}")

    if os.path.exists(risk_path):
        try:
            RISK_MODEL = joblib.load(risk_path)
        except Exception as e:
            print(f"Error: {e}")

load_models()

# --- ROUTES ---

@app.route("/")
def index():
    # Flask looks for index.html inside the 'static' folder
    return render_template("index.html")

@app.route("/api/detect_location", methods=["GET"])
def detect_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return jsonify({'state': g.state, 'country': g.country, 'latlng': g.latlng})
        return jsonify({'error': 'failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    if REGRESSORS is None:
        return jsonify({'error': 'Models not loaded'}), 500

    data = request.json or request.form
    user = {
        'Age': float(data.get('age', 30)),
        'Gender': data.get('gender', 'Male'),
        'BMI': float(data.get('bmi', 24.0)),
        'Daily_Steps': float(data.get('daily_steps', 3000)),
        'Exercise_Frequency': float(data.get('exercise_freq', 2)),
        'Sleep_Hours': float(data.get('sleep_hours', 7)),
        'Dietary_Habits': data.get('diet', 'Vegetarian'),
        'Chronic_Disease': data.get('chronic', 'None')
    }

    X_df = pd.DataFrame([user])
    preds = {}
    for k, model in REGRESSORS.items():
        preds[k] = float(model.predict(X_df)[0])

    # --- HEALTH SCORING LOGIC ---
    score = 100
    bmi = user['BMI']
    if bmi < 18.5 or bmi > 30: score -= 25
    elif bmi >= 25: score -= 10

    steps = user['Daily_Steps']
    if steps < 3000: score -= 20
    elif steps < 6000: score -= 10

    sleep = user['Sleep_Hours']
    if sleep < 6: score -= 15
    elif sleep < 7: score -= 5

    if user['Chronic_Disease'].lower() != 'none': score -= 20
    score = max(0, min(100, score))

    if score >= 85: health_label = "Excellent"
    elif score >= 70: health_label = "Good"
    elif score >= 50: health_label = "Average"
    elif score >= 30: health_label = "Poor"
    else: health_label = "Worst"

    return jsonify({
        'recommended': preds,
        'health_score': score,
        'health_label': health_label
    })

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json or request.form
    allergies = data.get('allergies', '')
    allergy_list = [a.strip() for a in allergies.split(',')] if allergies else []
    cuisine = data.get('cuisine')
    location = data.get('location')

    location_data = None
    if location:
        try:
            location_data = json.loads(location) if isinstance(location, str) else location
        except:
            location_data = {'state': location}

    rec = data.get('recommended')
    target = {
        'calories': float(rec.get('Recommended_Calories', 2000)),
        'protein': float(rec.get('Recommended_Protein', 100)),
        'carbs': float(rec.get('Recommended_Carbs', 250)),
        'fats': float(rec.get('Recommended_Fats', 70))
    }

    meals = load_meals()
    filtered = filter_meals(meals, allergies=allergy_list, cuisine_pref=cuisine, location=location_data)

    nearby_dishes = filtered[['Food Name', 'State', 'Type']].head(10).to_dict(orient='records')
    week_plan = plan_week(filtered, target)

    # Manual Week Plan Formatting
    out = []
    for day in week_plan:
        day_meals = []
        for m in day:
            day_meals.append({
                'name': m['Food Name'],
                'state': m['State'],
                'type': m['Type'],
                'calories': m['Total Calories'],
                'protein': m['Total Protein'],
                'carbs': m['Total Carbs'],
                'fats': m['Total Fats'],
            })
        out.append(day_meals)

    return jsonify({
        'week_plan': out,
        'nearby_dishes': nearby_dishes,
        'location': location_data
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
