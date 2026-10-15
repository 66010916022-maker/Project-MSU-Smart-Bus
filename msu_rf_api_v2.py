
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "msu_random_forest_model_simulated_v2.joblib"

model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return jsonify({
        "service": "MSU EV Bus ETA API V2",
        "status": "running",
        "model": MODEL_PATH.name,
        "data_note": "Model trained on simulated data, not real bus records"
    })

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a JSON object"
        }), 400

    required = [
        "distance_km",
        "speed_kmh",
        "hour",
        "traffic_index",
        "route_id",
        "day_of_week"
    ]

    missing = [key for key in required if key not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        distance = float(data["distance_km"])
        speed = float(data["speed_kmh"])
        hour = int(data["hour"])
        traffic = float(data["traffic_index"])

        if distance <= 0:
            raise ValueError("distance_km must be greater than 0")
        if speed <= 0:
            raise ValueError("speed_kmh must be greater than 0")
        if not 0 <= hour <= 23:
            raise ValueError("hour must be between 0 and 23")
        if traffic < 0:
            raise ValueError("traffic_index must be non-negative")

        features = pd.DataFrame([{
            "distance_km": distance,
            "speed_kmh": speed,
            "hour": hour,
            "traffic_index": traffic,
            "route_id": str(data["route_id"]),
            "day_of_week": str(data["day_of_week"])
        }])

        prediction = float(model.predict(features)[0])

        return jsonify({
            "status": "success",
            "route_id": str(data["route_id"]),
            "distance_km": distance,
            "predicted_travel_time_minutes": round(max(0, prediction), 2),
            "predicted_travel_time_seconds": round(max(0, prediction) * 60),
            "model": MODEL_PATH.name,
            "data_note": "Prediction based on simulated training data; not validated against real bus travel times"
        })

    except (ValueError, TypeError) as exc:
        return jsonify({
            "error": "Invalid input",
            "details": str(exc)
        }), 400

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5001, debug=False)
