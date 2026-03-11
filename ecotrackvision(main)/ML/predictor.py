"""
Water Forecast Predictor (Extended) — loads the saved model and predicts.
Features : household_size, income_level, property_type, dwelling_area_sqm, 
           has_garden, num_bathrooms, has_dishwasher, has_washing_machine, 
           occupants_children, water_price_per_m3, temperature_c
"""
import os
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

# Use resolve() for robust absolute path handling, with fallback for frozen modules on Windows
try:
    BASE_DIR = Path(__file__).resolve().parent
except (OSError, ValueError):
    BASE_DIR = Path(os.path.abspath(__file__)).parent

WATER_MODEL_PATH = str(BASE_DIR / 'models' / 'water_forecast_model.pkl')
ELEC_MODEL_PATH  = str(BASE_DIR / 'models' / 'electricity_forecast_model.pkl')

_water_model = None
_elec_model = None

def _fix_n_jobs(model):
    """Force model to use n_jobs=1 to avoid multiprocessing issues on Windows."""
    try:
        # If it's a Pipeline, fix the 'model' step
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            inner_model = model.named_steps['model']
            if hasattr(inner_model, 'n_jobs'):
                inner_model.n_jobs = 1
        # If it's a direct model
        elif hasattr(model, 'n_jobs'):
            model.n_jobs = 1
    except:
        pass
    return model

def _load_water_model():
    global _water_model
    if _water_model is None:
        if not os.path.exists(WATER_MODEL_PATH):
            raise FileNotFoundError(f"Water model not found at {WATER_MODEL_PATH}.")
        _water_model = joblib.load(WATER_MODEL_PATH)
        _fix_n_jobs(_water_model)
    return _water_model

def _load_elec_model():
    global _elec_model
    if _elec_model is None:
        if not os.path.exists(ELEC_MODEL_PATH):
            raise FileNotFoundError(f"Electricity model not found at {ELEC_MODEL_PATH}.")
        _elec_model = joblib.load(ELEC_MODEL_PATH)
        _fix_n_jobs(_elec_model)
    return _elec_model

def predict_water(
    household_size, income_level, property_type, dwelling_area_sqm,
    has_garden, num_bathrooms, has_dishwasher, has_washing_machine,
    occupants_children, water_price_per_m3, temperature_c
):
    """Predict daily water consumption (L)."""
    model = _load_water_model()
    data = {
        'household_size': [int(household_size)],
        'income_level': [income_level],
        'property_type': [property_type],
        'dwelling_area_sqm': [float(dwelling_area_sqm)],
        'has_garden': [int(has_garden)],
        'num_bathrooms': [int(num_bathrooms)],
        'has_dishwasher': [int(has_dishwasher)],
        'has_washing_machine': [int(has_washing_machine)],
        'occupants_children': [int(occupants_children)],
        'water_price_per_m3': [float(water_price_per_m3)],
        'temperature_c': [float(temperature_c)]
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]
    return round(float(max(prediction, 0)), 1)

def predict_electricity(
    household_size, income_level, property_type, dwelling_area_sqm,
    num_occupants_work_from_home, has_air_conditioner, has_electric_heating,
    has_ev, num_major_appliances, temperature_c, electricity_price_per_kwh
):
    """Predict daily electricity consumption (kWh)."""
    model = _load_elec_model()
    data = {
        'household_size': [int(household_size)],
        'income_level': [income_level],
        'property_type': [property_type],
        'dwelling_area_sqm': [float(dwelling_area_sqm)],
        'num_occupants_work_from_home': [int(num_occupants_work_from_home)],
        'has_air_conditioner': [int(has_air_conditioner)],
        'has_electric_heating': [int(has_electric_heating)],
        'has_ev': [int(has_ev)],
        'num_major_appliances': [int(num_major_appliances)],
        'temperature_c': [float(temperature_c)],
        'electricity_price_per_kwh': [float(electricity_price_per_kwh)]
    }
    X = pd.DataFrame(data)
    prediction = model.predict(X)[0]
    return round(float(max(prediction, 0)), 2)
