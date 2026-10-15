import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_FILE = "msu_bus_dataset.csv"

df = pd.read_csv(INPUT_FILE)

required = [
    "distance_km", "speed_kmh", "hour",
    "route_id", "day_of_week", "travel_time_minutes"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for c in ["distance_km", "speed_kmh", "hour", "travel_time_minutes"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=required).copy()

X = df[["distance_km", "speed_kmh", "hour", "route_id", "day_of_week"]]
y = df["travel_time_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     ["route_id", "day_of_week"]),
    ("num", "passthrough", ["distance_km", "speed_kmh", "hour"])
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=" * 50)
print("MSU SMART BUS - RANDOM FOREST")
print("=" * 50)
print(f"ข้อมูลทั้งหมด : {len(df)}")
print(f"Train        : {len(X_train)}")
print(f"Test         : {len(X_test)}")
print()
print(f"MAE  = {mae:.4f} นาที")
print(f"RMSE = {rmse:.4f} นาที")
print(f"R²   = {r2:.4f}")

joblib.dump(pipeline, "msu_random_forest_model.joblib")

result = X_test.copy()
result["actual_travel_time_minutes"] = y_test.values
result["predicted_travel_time_minutes"] = y_pred
result["absolute_error_minutes"] = np.abs(
    result["actual_travel_time_minutes"]
    - result["predicted_travel_time_minutes"]
)
result.to_csv("msu_rf_predictions.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([{
    "model": "Random Forest",
    "total_records": len(df),
    "train_records": len(X_train),
    "test_records": len(X_test),
    "MAE_minutes": mae,
    "RMSE_minutes": rmse,
    "R2": r2
}]).to_csv("msu_rf_metrics.csv", index=False, encoding="utf-8-sig")

print()
print("สร้างไฟล์:")
print("msu_random_forest_model.joblib")
print("msu_rf_predictions.csv")
print("msu_rf_metrics.csv")
