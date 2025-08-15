import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import geocoder
import json

# The dataset is now expected to be in a 'data' folder
MEALS_CSV = os.path.join("data", "meals_with_time_classification.csv")


def load_meals():
    meals = pd.read_csv(MEALS_CSV)
    num = ['Total Calories', 'Total Carbs', 'Total Fats', 'Total Protein', 'Total Sugar', 'Total Sodium']
    for c in num:
        meals[c] = pd.to_numeric(meals[c], errors='coerce').fillna(0)
    meals['Allergic Ingredients'] = meals['Allergic Ingredients'].fillna('None')
    return meals


def get_user_location():
    """
    Attempts to get the user's location using geocoder.ip('me') and falls back
    to a hardcoded location if the lookup fails.
    """
    try:
        g = geocoder.ip('me')
        if g.ok:
            # Check if the location data is reasonably specific
            if g.state and g.country:
                print(f"✅ Geocoder successfully detected location via IP: {g.state}, {g.country}")
                return {
                    'state': g.state,
                    'country': g.country,
                    'latlng': g.latlng
                }
            else:
                print("⚠️ Geocoder IP lookup succeeded but returned generic data. Falling back.")

    except Exception as e:
        print(f"❌ Error getting location via geocoder.ip: {e}")

    # Fallback location
    print("➡️ Falling back to default location: West Bengal, India.")
    return {
        'state': "West Bengal",
        'country': "India",
        'latlng': (22.9868, 87.8550)
    }


def filter_meals(meals, allergies=None, avoid_types=None, max_sodium=None,
                 max_sugar=None, cuisine_pref=None, location=None):
    df = meals.copy()
    if allergies:
        allergies = set([a.strip().lower() for a in allergies])

        def has_allergy(row):
            ingr = str(row['Allergic Ingredients']).lower()
            if ingr == 'none':
                return False
            for a in allergies:
                if a in ingr:
                    return True
            return False

        df = df[~df.apply(has_allergy, axis=1)]
    if avoid_types:
        df = df[~df['Type'].isin(avoid_types)]

    # Enhanced location filtering
    if location and 'state' in location:
        df = df[df['State'].str.contains(location['state'], case=False, na=False)]

    if cuisine_pref:
        df = df[df['State'].str.contains(cuisine_pref, case=False)]
    if max_sodium is not None:
        df = df[df['Total Sodium'] <= max_sodium]
    if max_sugar is not None:
        df = df[df['Total Sugar'] <= max_sugar]
    return df.reset_index(drop=True)


def score_meals(meals, target):
    nutrit = meals[['Total Calories', 'Total Protein', 'Total Carbs', 'Total Fats']].values
    nutrit_norm = nutrit / (np.maximum(nutrit.max(axis=0), 1))
    target_vec = np.array([target['calories'], target['protein'], target['carbs'], target['fats']], dtype=float)
    target_norm = target_vec / (np.maximum(target_vec.max(), 1))
    from sklearn.preprocessing import normalize
    s = cosine_similarity(nutrit_norm, target_norm.reshape(1, -1)).flatten()
    return s


def plan_week(meals, target_daily):
    meals = meals.copy()
    scores = score_meals(meals, target_daily)
    meals['score'] = scores

    morning_meals = meals[meals['Morning'] == 'Yes'].sort_values('score', ascending=False)
    afternoon_meals = meals[meals['Afternoon'] == 'Yes'].sort_values('score', ascending=False)
    night_meals = meals[meals['Night'] == 'Yes'].sort_values('score', ascending=False)

    selected_meals = set()
    week_plan = []

    for day in range(7):
        day_plan = []

        morning_candidate = morning_meals[~morning_meals['Food Name'].isin(selected_meals)].head(1)
        if not morning_candidate.empty:
            meal = morning_candidate.iloc[0].to_dict()
            day_plan.append(meal)
            selected_meals.add(meal['Food Name'])

        afternoon_candidate = afternoon_meals[~afternoon_meals['Food Name'].isin(selected_meals)].head(1)
        if not afternoon_candidate.empty:
            meal = afternoon_candidate.iloc[0].to_dict()
            day_plan.append(meal)
            selected_meals.add(meal['Food Name'])

        night_candidate = night_meals[~night_meals['Food Name'].isin(selected_meals)].head(1)
        if not night_candidate.empty:
            meal = night_candidate.iloc[0].to_dict()
            day_plan.append(meal)
            selected_meals.add(meal['Food Name'])

        week_plan.append(day_plan)

    return week_plan


if __name__ == "__main__":
    meals = load_meals()

    # Get precise user location
    user_location = get_user_location()
    print(f"Detected location: {user_location['state']}, {user_location['country']}")

    # Filter meals based on location and preferences
    filtered = filter_meals(meals, allergies=['Dairy'], location=user_location)

    # Display nearby dishes
    nearby_dishes = filtered[['Food Name', 'State', 'Type']].head(10)
    print("\nNearby dishes:")
    print(nearby_dishes)

    # Create meal plan
    target = {'calories': 2000, 'protein': 100, 'carbs': 250, 'fats': 70}
    week = plan_week(filtered, target)

    print("\nWeekly Meal Plan:")
    for d, day in enumerate(week):
        print(f"\nDay {d + 1}")
        for m in day:
            print(f" - {m['Food Name']} ({m['Type']}): {m['Total Calories']} cal")