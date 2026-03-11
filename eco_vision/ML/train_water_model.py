"""
EcoTrack Vision — Water Consumption Forecasting (Proper ML Pipeline)
======================================================================
Dataset  : ml/water_consumption_forecast.csv  (1096 rows, 3 years)
Features : Household_Size, Temperature_C, Is_Weekend, Month,
           Day_of_Week, Season, Prev_Day_Usage_L
Target   : Daily_Usage_L

Algorithms:
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Outputs:
  ml/models/water_forecast_model.pkl
  ml/models/training_results.json
  ml/models/evaluation_report.txt
  ml/models/model_comparison.png
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, json, datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, 'water_consumption_forecast.csv')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
MODEL_PATH   = os.path.join(MODEL_DIR, 'water_forecast_model.pkl')
RESULTS_PATH = os.path.join(MODEL_DIR, 'training_results.json')
REPORT_PATH  = os.path.join(MODEL_DIR, 'evaluation_report.txt')
GRAPH_PATH   = os.path.join(MODEL_DIR, 'model_comparison.png')

os.makedirs(MODEL_DIR, exist_ok=True)

SEP = "=" * 68

# ── 1. Load ────────────────────────────────────────────────────────────────
print(SEP)
print("  EcoTrack Vision — Water Consumption Forecast Training")
print(SEP)

df = pd.read_csv(DATA_PATH)
print(f"\n[OK] Dataset : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"     File    : {DATA_PATH}")

# ── 2. Features & Target ───────────────────────────────────────────────────
FEATURES = [
    'Household_Size', 'Temperature_C', 'Is_Weekend',
    'Month', 'Day_of_Week', 'Season', 'Prev_Day_Usage_L'
]
TARGET = 'Daily_Usage_L'

X = df[FEATURES]
y = df[TARGET]

print(f"\n[OK] Features ({len(FEATURES)}) : {FEATURES}")
print(f"     Target             : {TARGET}")
print(f"     Mean  = {y.mean():.1f} L  |  Std = {y.std():.1f} L"
      f"  |  Range = [{y.min():.0f}, {y.max():.0f}] L")

# ── 3. Train / Test Split (80 / 20) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n[OK] Train : {len(X_train)} samples  |  Test : {len(X_test)} samples")

# ── 4. Define 3 Models ────────────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ]),
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestRegressor(
            n_estimators=200, max_depth=10,
            random_state=42, n_jobs=-1
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        )),
    ]),
}

# ── 5. Train, Evaluate & Compare ───────────────────────────────────────────
results       = {}
best_name     = None
best_r2       = -999
best_pipeline = None

print(f"\n{'-'*68}")
print(f"  {'Algorithm':<22}  {'MAE (L)':>9}  {'RMSE (L)':>9}  {'R2':>8}  {'CV R2 (5-fold)':>16}")
print(f"{'-'*68}")

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(pipeline, X, y, cv=5, scoring='r2').mean()

    results[name] = {
        "mae":   round(mae,  2),
        "rmse":  round(rmse, 2),
        "r2":    round(r2,   4),
        "cv_r2": round(cv,   4),
    }

    print(f"  {name:<22}  {mae:>9.2f}  {rmse:>9.2f}  {r2:>8.4f}  {cv:>16.4f}")

    if r2 > best_r2:
        best_r2, best_name, best_pipeline = r2, name, pipeline

print(f"{'-'*68}")
print(f"\n  [BEST] {best_name}  ->  R2 = {best_r2:.4f}\n")

# ── 6. Save Best Model ────────────────────────────────────────────────────
joblib.dump(best_pipeline, MODEL_PATH)
print(f"[SAVED] Model  -> {MODEL_PATH}")

# ── 7. Save JSON ──────────────────────────────────────────────────────────
summary = {
    "best_model": best_name,
    "best_r2":    round(best_r2, 4),
    "features":   FEATURES,
    "target":     TARGET,
    "dataset":    "water_consumption_forecast.csv",
    "results":    results,
}
with open(RESULTS_PATH, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"[SAVED] JSON   -> {RESULTS_PATH}")

# ── 8. Evaluation TXT Report ──────────────────────────────────────────────
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    "=" * 70,
    "  EcoTrack Vision - Water Consumption Forecast Evaluation Report",
    f"  Generated  : {now}",
    "=" * 70,
    "",
    f"  Dataset        : water_consumption_forecast.csv",
    f"  Total Rows     : {len(df)}",
    f"  Date Range     : {df['Date'].iloc[0]}  to  {df['Date'].iloc[-1]}",
    f"  Train Samples  : {len(X_train)}  (80%)",
    f"  Test  Samples  : {len(X_test)}  (20%)",
    f"  Target         : {TARGET}",
    f"  Target Mean    : {y.mean():.2f} L",
    f"  Target Std Dev : {y.std():.2f} L",
    f"  Target Range   : [{y.min():.0f}, {y.max():.0f}] L",
    "",
    f"  Features ({len(FEATURES)}):",
]
feat_desc = {
    'Household_Size':   'Number of people in household',
    'Temperature_C':    'Ambient temperature in Celsius',
    'Is_Weekend':       '1 if Saturday/Sunday, else 0',
    'Month':            'Calendar month (1-12)',
    'Day_of_Week':      'Day of week (0=Mon, 6=Sun)',
    'Season':           '0=Winter 1=Spring 2=Summer 3=Autumn',
    'Prev_Day_Usage_L': 'Previous day total consumption (lag-1)',
}
for f_name in FEATURES:
    lines.append(f"    * {f_name:<22} : {feat_desc[f_name]}")

lines += [
    "",
    "-" * 70,
    f"  {'Algorithm':<24} {'MAE (L)':>9} {'RMSE (L)':>10} {'R2':>9} {'CV R2':>9}",
    "-" * 70,
]
for name, m in results.items():
    star = "  <-- BEST" if name == best_name else ""
    lines.append(
        f"  {name:<24} {m['mae']:>9.2f} {m['rmse']:>10.2f} {m['r2']:>9.4f} {m['cv_r2']:>9.4f}{star}"
    )
lines += [
    "-" * 70,
    "",
    f"  Best Model : {best_name}",
    f"  Best R2    : {best_r2:.4f}",
    f"  Saved to   : {MODEL_PATH}",
    "",
    "=" * 70,
    "  Metric Reference",
    "=" * 70,
    "  MAE    : Mean Absolute Error in Liters (avg daily prediction error)",
    "  RMSE   : Root Mean Squared Error in Liters (punishes large errors more)",
    "  R2     : Coefficient of Determination  (1.0 = perfect fit)",
    "  CV R2  : 5-Fold Cross-Validation R2   (model generalization ability)",
    "",
]

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"[SAVED] TXT    -> {REPORT_PATH}")

# ── 9. Model Comparison Graph ─────────────────────────────────────────────
model_names = list(results.keys())
mae_vals    = [results[n]['mae']   for n in model_names]
rmse_vals   = [results[n]['rmse']  for n in model_names]
r2_vals     = [results[n]['r2']    for n in model_names]
cv_vals     = [results[n]['cv_r2'] for n in model_names]

COLORS = ['#00b894', '#0984e3', '#e17055']
short  = [n.replace(" ", "\n") for n in model_names]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Water Consumption Forecast — Model Comparison',
    fontsize=14, fontweight='bold', y=1.03
)

metrics = [
    ("MAE — Liters\n(lower is better)",  mae_vals,  COLORS[0]),
    ("RMSE — Liters\n(lower is better)", rmse_vals, COLORS[1]),
    ("R² Score\n(higher is better)",     r2_vals,   COLORS[2]),
]

for ax, (title, vals, color) in zip(axes, metrics):
    bars = ax.bar(short, vals, color=color, alpha=0.82,
                  edgecolor='white', linewidth=1.5, width=0.45)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.02,
            f'{val:.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    best_idx = vals.index(min(vals)) if 'lower' in title else vals.index(max(vals))
    bars[best_idx].set_edgecolor('#2d3436')
    bars[best_idx].set_linewidth(3)
    bars[best_idx].set_alpha(1.0)
    ax.text(
        bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
        -max(vals) * 0.12,
        'BEST', ha='center', fontsize=8, color='#2d3436', fontweight='bold'
    )

fig.text(
    0.5, -0.05,
    "5-Fold CV R2:   " + "   |   ".join(
        [f"{n}: {results[n]['cv_r2']:.4f}" for n in model_names]
    ),
    ha='center', fontsize=9.5, color='#636e72', style='italic'
)

plt.tight_layout()
plt.savefig(GRAPH_PATH, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[SAVED] Graph  -> {GRAPH_PATH}")

print(f"\n{SEP}")
print("  Training & Evaluation Complete!")
print(SEP)
