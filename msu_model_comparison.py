import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

NUM = ["distance_km", "speed_kmh", "hour"]
CAT = ["route_id", "day_of_week"]
TARGET = "travel_time_minutes"
REQUIRED = NUM + CAT + [TARGET]

def preprocessor(scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUM),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CAT)
    ])

def metrics(name, actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {"model": name, "n": 0, "MAE_min": np.nan, "RMSE_min": np.nan,
                "R2": np.nan, "within_1_min_pct": np.nan, "within_2_min_pct": np.nan}
    err = np.abs(actual[valid] - predicted[valid])
    return {
        "model": name, "n": int(valid.sum()),
        "MAE_min": mean_absolute_error(actual[valid], predicted[valid]),
        "RMSE_min": np.sqrt(mean_squared_error(actual[valid], predicted[valid])),
        "R2": r2_score(actual[valid], predicted[valid]) if valid.sum() >= 2 else np.nan,
        "within_1_min_pct": 100 * np.mean(err <= 1),
        "within_2_min_pct": 100 * np.mean(err <= 2)
    }

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("MSU_EV_BUS_TRAVEL_HISTORY.csv")
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}. วาง CSV ไว้โฟลเดอร์เดียวกัน หรือระบุ path หลังชื่อสคริปต์")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("คอลัมน์ที่ขาด: " + ", ".join(missing) +
                         "\nคอลัมน์ในไฟล์: " + ", ".join(map(str, df.columns)))
    df = df[REQUIRED].copy()
    for c in NUM + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f"มีข้อมูลใช้ได้เพียง {len(df)} แถว ต้องมีอย่างน้อย 10 แถว")
    X, y = df[NUM + CAT], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    results = []
    d, s = X_test["distance_km"].to_numpy(), X_test["speed_kmh"].to_numpy()
    base = np.full(len(X_test), np.nan)
    ok = np.isfinite(d) & np.isfinite(s) & (d >= 0) & (s > 0)
    base[ok] = d[ok] / s[ok] * 60
    results.append(metrics("Baseline (distance/speed)", y_test, base))

    models = {
        "Linear Regression": Pipeline([
            ("prep", preprocessor(scale=True)), ("model", LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ("prep", preprocessor()), ("model", DecisionTreeRegressor(random_state=42, min_samples_leaf=2))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2, n_jobs=-1))
        ])
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(metrics(name, y_test, model.predict(X_test)))
        if name == "Random Forest":
            joblib.dump(model, "msu_random_forest_model.joblib")
            print("บันทึกโมเดล: msu_random_forest_model.joblib")
    result = pd.DataFrame(results).sort_values("MAE_min")
    print("\nผลเปรียบเทียบโมเดล (ใช้ชุดทดสอบเดียวกัน)")
    print(result.round(4).to_string(index=False))
    result.to_csv("msu_model_comparison_results.csv", index=False, encoding="utf-8-sig")
    print("\nบันทึกผลแล้ว: msu_model_comparison_results.csv")
    print(f"จำนวนข้อมูล {len(df)} | Train {len(X_train)} | Test {len(X_test)}")
    print("หมายเหตุ: คะแนนนี้ประเมินเฉพาะข้อมูลชุดนี้ ข้อมูลจำลองไม่ยืนยันความแม่นยำกับรถจริง")

if __name__ == "__main__":
    main()
