from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = r"C:\MSU_Bus_ML\msu_random_forest_model.joblib"

model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    distance_km = float(data.get("distance_km", 0))
    speed_kmh = float(data.get("speed_kmh", 0))
    hour = int(data.get("hour", 0))
    route_id = data.get("route_id", "")
    day_of_week = data.get("day_of_week", "")

    input_data = pd.DataFrame([{
        "distance_km": distance_km,
        "speed_kmh": speed_kmh,
        "hour": hour,
        "route_id": route_id,
        "day_of_week": day_of_week
    }])

    prediction = model.predict(input_data)[0]

    prediction = max(1, float(prediction))

    return jsonify({
        "predicted_eta_minutes": round(prediction, 2),
        "distance_km": distance_km,
        "speed_kmh": speed_kmh,
        "hour": hour,
        "route_id": route_id,
        "day_of_week": day_of_week
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)