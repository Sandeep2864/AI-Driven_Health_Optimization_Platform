from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json
import geocoder
import sys

# --- PATHING CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Import logic from recommender.py
try:
    from recommender import load_meals, filter_meals, plan_week
except ImportError as e:
    print(f"❌ Critical Error: Could not find recommender.py in root. {e}")

app = Flask(__name__, 
            template_folder="static", 
            static_folder="static",
            static_url_path="/static")
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
            print("✅ Regressors loaded successfully")
        except Exception as e:
            print(f"❌ Error loading regressors: {e}")

    if os.path.exists(risk_path):
        try:
            RISK_MODEL = joblib.load(risk_path)
            print("✅ Risk model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading risk model: {e}")

# Pre-load models on startup
load_models()

# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/detect_location", methods=["GET"])
def detect_location():
    try:
        g = geocoder.ip('me')
        if g.ok:
            return jsonify({
                'state': g.state,
                'country': g.country,
                'latlng': g.latlng
            })
        return jsonify({'error': 'Location detection failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    if REGRESSORS is None:
        return jsonify({'error': 'Machine Learning models not found on server'}), 500

    data = request.json or request.form
    user = {
        'Age': float(data.get('age', 30)),
        'Gender': data.get('gender', 'Other'),
        'BMI': float(data.get('bmi', 24.0)),
        'Daily_Steps': float(data.get('daily_steps', 3000)),
        'Exercise_Frequency': float(data.get('exercise_freq', 2)),
        'Sleep_Hours': float(data.get('sleep_hours', 7)),
        'Dietary_Habits': data.get('diet', 'Regular'),
        'Chronic_Disease': data.get('chronic', 'None')
    }

    X_df = pd.DataFrame([user])
    preds = {}
    for k, model in REGRESSORS.items():
        preds[k] = float(model.predict(X_df)[0])

    # --- FULL SCORING LOGIC (Restored) ---
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
    cuisine = data.get('cuisine', None)
    location = data.get('location', None)

    location_data = None
    if location:
        try:
            location_data = json.loads(location) if isinstance(location, str) else location
        except:
            location_data = {'state': location}

    rec = data.get('recommended')
    if not rec:
        return jsonify({'error': 'Missing targets. Run prediction first.'}), 400

    target = {
        'calories': float(rec.get('Recommended_Calories', rec.get('RecommendedCalories', 2000))),
        'protein': float(rec.get('Recommended_Protein', 100)),
        'carbs': float(rec.get('Recommended_Carbs', 250)),
        'fats': float(rec.get('Recommended_Fats', 70))
    }

    meals = load_meals()
    filtered = filter_meals(meals, allergies=allergy_list, cuisine_pref=cuisine, location=location_data)

    nearby_dishes = filtered[['Food Name', 'State', 'Type']].head(10).to_dict(orient='records')
    week_plan = plan_week(filtered, target)

    # --- FULL FORMATTING LOGIC (Restored) ---
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
