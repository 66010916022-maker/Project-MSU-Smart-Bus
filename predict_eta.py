import sys
import json
import joblib
import pandas as pd

MODEL_PATH = r"C:\MSU_Bus_ML\msu_random_forest_model.joblib"

# รับข้อมูลจาก n8n
distance_km = float(sys.argv[1])
speed_kmh = float(sys.argv[2])
hour = int(sys.argv[3])
route_id = sys.argv[4]
day_of_week = sys.argv[5]

# โหลดโมเดล
model = joblib.load(MODEL_PATH)

# สร้างข้อมูลสำหรับทำนาย
data = pd.DataFrame([{
    "distance_km": distance_km,
    "speed_kmh": speed_kmh,
    "hour": hour,
    "route_id": route_id,
    "day_of_week": day_of_week
}])

# ทำนาย
prediction = model.predict(data)[0]

# ไม่ให้ ETA ต่ำกว่า 1 นาที
prediction = max(1, float(prediction))

result = {
    "predicted_eta_minutes": round(prediction, 2),
    "distance_km": distance_km,
    "speed_kmh": speed_kmh,
    "route_id": route_id,
    "day_of_week": day_of_week
}

print(json.dumps(result, ensure_ascii=False))