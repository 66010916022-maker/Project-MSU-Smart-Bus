from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ========================================
# โหลดโมเดล Random Forest
# ========================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "msu_random_forest_model.joblib"
)

model = joblib.load(MODEL_PATH)

print("========================================")
print("MSU SMART BUS - RANDOM FOREST API")
print("========================================")
print("Model loaded successfully")
print("API: http://localhost:5000/predict")
print("========================================")


# ========================================
# API Predict
# ========================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        # -------------------------------
        # ตรวจสอบข้อมูล
        # -------------------------------

        required_fields = [
            "distance_km",
            "speed_kmh",
            "hour",
            "route_id",
            "day_of_week"
        ]

        for field in required_fields:

            if field not in data:

                return jsonify({
                    "error": f"Missing field: {field}"
                }), 400


        # -------------------------------
        # สร้าง DataFrame
        # -------------------------------

        input_data = pd.DataFrame([{

            "distance_km": float(data["distance_km"]),

            "speed_kmh": float(data["speed_kmh"]),

            "hour": int(data["hour"]),

            "route_id": str(data["route_id"]),

            "day_of_week": str(data["day_of_week"])

        }])


        # -------------------------------
        # Random Forest Prediction
        # -------------------------------

        prediction = model.predict(input_data)[0]

        prediction = max(0, float(prediction))


        # -------------------------------
        # ส่งผลลัพธ์
        # -------------------------------

        return jsonify({

            "predicted_eta_minutes":
                round(prediction, 2),

            "distance_km":
                float(data["distance_km"]),

            "speed_kmh":
                float(data["speed_kmh"]),

            "hour":
                int(data["hour"]),

            "route_id":
                str(data["route_id"]),

            "day_of_week":
                str(data["day_of_week"])

        })


    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500


# ========================================
# Run API
# ========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )