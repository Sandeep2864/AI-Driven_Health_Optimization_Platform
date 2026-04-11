from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import json
import geocoder

# --- PATHING CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Import logic from recommender.py
try:
    from recommender import load_meals, filter_meals, plan_week
except ImportError as e:
    print(f"❌ Critical Error: Could not find recommender.py. {e}")

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
            print("✅ Regressors loaded")
        except Exception as e:
            print(f"❌ Regressor load error: {e}")

    if os.path.exists(risk_path):
        try:
            RISK_MODEL = joblib.load(risk_path)
            print("✅ Risk model loaded")
        except Exception as e:
            print(f"❌ Risk model load error: {e}")

# Load models on server start
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
            return jsonify({'state': g.state, 'country': g.country, 'latlng': g.latlng})
        return jsonify({'error': 'Location detection failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    if REGRESSORS is None:
        return jsonify({'error': 'ML models not found on server'}), 500

    data = request.json or request.form
    user_data = {
        'Age': float(data.get('age', 30)),
        'Gender': data.get('gender', 'Male'),
        'BMI': float(data.get('bmi', 24.0)),
        'Daily_Steps': float(data.get('daily_steps', 3000)),
        'Exercise_Frequency': float(data.get('exercise_freq', 2)),
        'Sleep_Hours': float(data.get('sleep_hours', 7)),
        'Dietary_Habits': data.get('diet', 'Vegetarian'),
        'Chronic_Disease': data.get('chronic', 'None')
    }

    X_df = pd.DataFrame([user_data])
    preds = {k: float(model.predict(X_df)[0]) for k, model in REGRESSORS.items()}

    # Calculate Health Label
    risk_val = "Low"
    if RISK_MODEL:
        risk_val = RISK_MODEL.predict(X_df)[0]

    return jsonify({
        'recommended': preds,
        'health_label': risk_val
    })

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json or request.form
    rec = data.get('recommended', {})
    
    target = {
        'calories': float(rec.get('Recommended_Calories', 2000)),
        'protein': float(rec.get('Recommended_Protein', 100)),
        'carbs': float(rec.get('Recommended_Carbs', 250)),
        'fats': float(rec.get('Recommended_Fats', 70))
    }

    meals = load_meals()
    filtered = filter_meals(
        meals, 
        allergies=data.get('allergies', ''),
        cuisine_pref=data.get('cuisine'),
        location={'state': data.get('location')}
    )

    week_plan_raw = plan_week(filtered, target)
    
    # Format for JSON response
    formatted_plan = []
    for day in week_plan_raw:
        formatted_plan.append([{
            'name': m['Food Name'],
            'calories': m['Total Calories'],
            'protein': m['Total Protein'],
            'type': m['Type']
        } for m in day])

    return jsonify({
        'week_plan': formatted_plan,
        'nearby_dishes': filtered[['Food Name', 'State']].head(5).to_dict(orient='records')
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
