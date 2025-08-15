from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
from recommender import load_meals, filter_meals, plan_week
import geocoder

app = Flask(__name__)
MODEL_DIR = "models"

# load models (trained earlier)
REGRESSORS = None
RISK_MODEL = None


def load_models():
    global REGRESSORS, RISK_MODEL
    REGRESSORS = joblib.load(os.path.join(MODEL_DIR, "regressors.pkl"))
    RISK_MODEL = joblib.load(os.path.join(MODEL_DIR, "risk_model.pkl"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect_location", methods=["GET"])
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


@app.route("/predict", methods=["POST"])
def predict():
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

    # Predict nutrition recommendations
    preds = {}
    for k, model in REGRESSORS.items():
        preds[k] = float(model.predict(X_df)[0])

    # Compute a dynamic health score (0-100)
    score = 100

    # BMI scoring
    bmi = user['BMI']
    if bmi < 18.5 or bmi > 30:
        score -= 25
    elif bmi >= 25:
        score -= 10

    # Steps scoring
    steps = user['Daily_Steps']
    if steps < 3000:
        score -= 20
    elif steps < 6000:
        score -= 10

    # Sleep scoring
    sleep = user['Sleep_Hours']
    if sleep < 6:
        score -= 15
    elif sleep < 7:
        score -= 5

    # Chronic disease penalty
    if user['Chronic_Disease'].lower() != 'none':
        score -= 20

    # Bound score
    score = max(0, min(100, score))

    # Map to category
    if score >= 85:
        health_label = "Excellent"
    elif score >= 70:
        health_label = "Good"
    elif score >= 50:
        health_label = "Average"
    elif score >= 30:
        health_label = "Poor"
    else:
        health_label = "Worst"

    return jsonify({
        'recommended': preds,
        'health_score': score,
        'health_label': health_label
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json or request.form
    allergies = data.get('allergies', '')
    allergy_list = [a.strip() for a in allergies.split(',')] if allergies else []
    cuisine = data.get('cuisine', None)
    location = data.get('location', None)

    # Parse location if available
    location_data = None
    if location:
        try:
            location_data = json.loads(location)
        except:
            location_data = {'state': location}

    # assume client first called /predict to get recommended calories/protein etc.
    rec = data.get('recommended')  # dict or JSON string
    if not rec:
        return jsonify(
            {'error': 'No recommended targets provided. Call /predict first or include recommended targets.'}), 400
    target = {
        'calories': float(rec.get('Recommended_Calories', rec.get('RecommendedCalories', 2000))),
        'protein': float(rec.get('Recommended_Protein', 100)),
        'carbs': float(rec.get('Recommended_Carbs', 250)),
        'fats': float(rec.get('Recommended_Fats', 70))
    }

    meals = load_meals()
    filtered = filter_meals(meals, allergies=allergy_list, cuisine_pref=cuisine, location=location_data)

    # Get nearby dishes for display
    nearby_dishes = filtered[['Food Name', 'State', 'Type']].head(10).to_dict(orient='records')

    week_plan = plan_week(filtered, target)
    # Convert to JSON-friendly structure with all nutrition data
    out = []
    for day in week_plan:
        day_meals = []
        for m in day:
            meal_data = {
                'name': m['Food Name'],
                'state': m['State'],
                'type': m['Type'],
                'calories': m['Total Calories'],
                'protein': m['Total Protein'],
                'carbs': m['Total Carbs'],
                'fats': m['Total Fats'],
            }
            day_meals.append(meal_data)
        out.append(day_meals)
    return jsonify({
        'week_plan': out,
        'nearby_dishes': nearby_dishes,
        'location': location_data
    })


if __name__ == "__main__":
    load_models()
    app.run(debug=True, port=5000)