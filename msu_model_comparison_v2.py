
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

CSV_FILE = "MSU_EV_BUS_SYNTHETIC_TRAVEL_HISTORY_V2.csv"

NUMERIC = ["distance_km", "speed_kmh", "hour", "traffic_index"]
CATEGORICAL = ["route_id", "day_of_week"]
TARGET = "travel_time_minutes"


def make_preprocessor(scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL)
    ])


def evaluate(name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = np.abs(actual - predicted)

    return {
        "model": name,
        "test_rows": len(actual),
        "MAE_minutes": mean_absolute_error(actual, predicted),
        "RMSE_minutes": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "within_1_min_percent": np.mean(errors <= 1) * 100,
        "within_2_min_percent": np.mean(errors <= 2) * 100
    }


def main():
    csv_path = Path(CSV_FILE)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ CSV: {csv_path.resolve()}\n"
            "กรุณาวาง CSV ไว้ใน C:\\MSU_Bus_ML"
        )

    df = pd.read_csv(csv_path)

    if "source_type" not in df.columns:
        raise ValueError("CSV ต้องมีคอลัมน์ source_type")

    sources = set(df["source_type"].dropna().astype(str).unique())
    if sources != {"SIMULATED_V2"}:
        raise ValueError(f"source_type ไม่ตรงกับ SIMULATED_V2: {sources}")

    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV ขาดคอลัมน์: {missing}")

    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).copy()
    df = df[(df.distance_km > 0) & (df.speed_kmh > 0)].reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"มีข้อมูลใช้ได้เพียง {len(df)} แถว ต้องมีอย่างน้อย 50 แถว")

    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = X_test["distance_km"] / X_test["speed_kmh"] * 60
    results = [evaluate("Baseline distance/speed", y_test, baseline)]

    models = {
        "Linear Regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("model", LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ("prep", make_preprocessor()),
            ("model", DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=5, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=200, min_samples_leaf=3,
                random_state=42, n_jobs=-1
            ))
        ])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(evaluate(name, y_test, predictions))

        if name == "Random Forest":
            import joblib
            joblib.dump(model, "msu_random_forest_model_simulated_v2.joblib")

    result_df = pd.DataFrame(results).sort_values("MAE_minutes")
    print("\n=== MODEL COMPARISON: SIMULATED V2 ===")
    print(result_df.round(4).to_string(index=False))
    print(f"\nข้อมูลที่ใช้ได้: {len(df)} rows")
    print(f"Training: {len(X_train)} rows")
    print(f"Testing: {len(X_test)} rows")

    result_df.to_csv(
        "msu_model_comparison_results_v2.csv",
        index=False, encoding="utf-8-sig"
    )
    print("\nบันทึกผลแล้ว: msu_model_comparison_results_v2.csv")
    print("โมเดล V2: msu_random_forest_model_simulated_v2.joblib")
    print("หมายเหตุ: ผลนี้ประเมินจากข้อมูลจำลอง ไม่ใช่ข้อมูลรถจริง")


if __name__ == "__main__":
    main()
