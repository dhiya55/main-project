"""
EcoTrack Vision — Electricity Consumption Training
==================================================
Dataset  : ml/household_electricity_consumption.csv
Features : household_size, income_level (encoded), property_type (encoded), 
           dwelling_area_sqm, num_occupants_work_from_home, has_air_conditioner,
           has_electric_heating, has_ev, num_major_appliances, temperature_c,
           electricity_price_per_kwh
Target   : daily_consumption_kwh
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, 'household_electricity_consumption.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'electricity_forecast_model.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'electricity_training_results.json')
REPORT_PATH  = os.path.join(MODEL_DIR, 'electricity_evaluation_report.txt')
GRAPH_PATH   = os.path.join(MODEL_DIR, 'electricity_model_comparison.png')

os.makedirs(MODEL_DIR, exist_ok=True)

SEP = "=" * 68

# ── 1. Load Data ───────────────────────────────────────────────────────────
print(SEP)
print("  EcoTrack Vision — Electricity Consumption Training")
print(SEP)

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found: {DATA_PATH}")
    exit(1)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset : {df.shape[0]} rows x {df.shape[1]} columns")

# ── 2. Preprocessing ───────────────────────────────────────────────────────
# Drop ID
if 'household_id' in df.columns:
    df = df.drop(columns=['household_id'])

TARGET = 'daily_consumption_kwh'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Define feature types
numeric_features = [
    'household_size', 'dwelling_area_sqm', 'num_occupants_work_from_home',
    'num_major_appliances', 'temperature_c', 'electricity_price_per_kwh'
]
categorical_features = ['income_level', 'property_type']
binary_features = ['has_air_conditioner', 'has_electric_heating', 'has_ev']

print(f"[OK] Features: {list(X.columns)}")
print(f"     Target  : {TARGET}")

# ── 3. Split ───────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ── 4. Build Pipeline ──────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

def create_pipeline(model):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

models = {
    "Linear Regression": create_pipeline(LinearRegression()),
    "Random Forest": create_pipeline(RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )),
    "Gradient Boosting": create_pipeline(GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
    ))
}

# ── 5. Train & Evaluate ────────────────────────────────────────────────────
results = {}
best_name = None
best_r2 = -np.inf
best_pipeline = None

print(f"\n{'-'*68}")
print(f"  {'Algorithm':<22}  {'MAE (kWh)':>9}  {'RMSE (kWh)':>10}  {'R2':>8}  {'CV R2':>10}")
print(f"{'-'*68}")

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Simple CV
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    
    results[name] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "cv_r2": round(cv_mean, 4)
    }
    
    print(f"  {name:<22}  {mae:>9.2f}  {rmse:>10.2f}  {r2:>8.4f}  {cv_mean:>10.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_name = name
        best_pipeline = pipeline

print(f"{'-'*68}")
print(f"\n  [BEST] {best_name} -> R2 = {best_r2:.4f}")

# ── 6. Save Artifacts ──────────────────────────────────────────────────────
joblib.dump(best_pipeline, MODEL_PATH)

summary = {
    "best_model": best_name,
    "best_r2": round(best_r2, 4),
    "features": list(X.columns),
    "target": TARGET,
    "dataset": "household_electricity_consumption.csv",
    "results": results,
    "timestamp": datetime.datetime.now().isoformat()
}

with open(RESULTS_PATH, 'w') as f:
    json.dump(summary, f, indent=4)

# TXT Report
with open(REPORT_PATH, 'w') as f:
    f.write(f"EcoTrack Vision - Electricity Consumption Report\n")
    f.write(f"Generated: {datetime.datetime.now()}\n")
    f.write(f"{'='*50}\n")
    f.write(f"Dataset: {DATA_PATH}\n")
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Best R2: {best_r2:.4f}\n\n")
    f.write(f"{'Algorithm':<20} {'MAE':>10} {'RMSE':>10} {'R2':>10}\n")
    for name, res in results.items():
        f.write(f"{name:<20} {res['mae']:>10.2f} {res['rmse']:>10.2f} {res['r2']:>10.4f}\n")

# Visualization
model_names = list(results.keys())
r2_vals = [results[n]['r2'] for n in model_names]
mae_vals = [results[n]['mae'] for n in model_names]

fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Models')
ax1.set_ylabel('R2 Score', color=color)
ax1.bar(model_names, r2_vals, color=color, alpha=0.6, label='R2 Score')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('MAE (kWh)', color=color)
ax2.plot(model_names, mae_vals, color=color, marker='o', label='MAE')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Electricity Model Comparison')
fig.tight_layout()
plt.savefig(GRAPH_PATH)
plt.close()

print(f"\n[OK] Training completed successfully.")
print(f"     Model saved to: {MODEL_PATH}")
