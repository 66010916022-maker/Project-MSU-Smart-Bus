import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

# ==========================================
# MSU Smart Bus - Random Forest ETA Model
# ==========================================

CSV_FILE = "msu_bus_dataset.csv"
MODEL_FILE = "msu_random_forest_model.joblib"

# 1) Load dataset
df = pd.read_csv(CSV_FILE)

required = [
    "distance_km",
    "speed_kmh",
    "hour",
    "route_id",
    "day_of_week",
    "travel_time_minutes",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required].copy()

# 2) Clean numeric fields
for col in ["distance_km", "speed_kmh", "hour", "travel_time_minutes"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required)

if len(df) < 10:
    raise ValueError(f"Only {len(df)} usable rows. Need at least 10 rows.")

# 3) Features and target
X = df[
    ["distance_km", "speed_kmh", "hour", "route_id", "day_of_week"]
]
y = df["travel_time_minutes"]

# 4) Train/Test = 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 5) Encode categorical variables
categorical = ["route_id", "day_of_week"]
numeric = ["distance_km", "speed_kmh", "hour"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

# 6) Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# 7) Predict test set
y_pred = pipeline.predict(X_test)

# 8) Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MSU BUS RANDOM FOREST ==========")
print(f"Total usable records : {len(df)}")
print(f"Train records        : {len(X_train)}")
print(f"Test records         : {len(X_test)}")
print("-------------------------------------------")
print(f"MAE                  : {mae:.4f} minutes")
print(f"RMSE                 : {rmse:.4f} minutes")
print(f"R²                   : {r2:.4f}")
print("===========================================\n")

# 9) Show prediction comparison
result = X_test.copy()
result["actual_minutes"] = y_test.values
result["predicted_minutes"] = np.round(y_pred, 3)
result["error_minutes"] = np.round(
    result["actual_minutes"] - result["predicted_minutes"], 3
)

print("Prediction results:")
print(result.to_string(index=False))

# 10) Save model
joblib.dump(pipeline, MODEL_FILE)
print(f"\nModel saved as: {MODEL_FILE}")
