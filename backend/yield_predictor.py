import os
import logging
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "maize":    "maize_rf_model.pkl",
    "oats":     "oats_rf_model.pkl",
    "soybeans": "soybean_rf_model.pkl",
    "wheat":    "wheat_rf_model.pkl",
}

# Ohio historical averages used as defaults when live weather is unavailable
WEATHER_DEFAULTS = {
    "maize": {
        "opt_temp_min": 16.33, "opt_temp_max": 25.0, "opt_precip": 113.33,
        "stage_duration_days": 85.33, "temp_deviation": 47.75,
        "precip_deviation": 106.45, "water_stress_indicator": 0.624,
        "total_extreme_weather_days": 10.48, "hail_events": 0.5,
        "tornado_events": 0.05, "high_wind_events": 1.2, "thunder_events": 3.1,
        "GDD": 34.10, "precip_total": 6.89, "precip_avg": 0.23,
        "precip_max": 1.45, "Temp_avg": 68.42, "TempMAX_avg": 78.76,
        "TempMIN_avg": 58.19, "Snow_total": 0.0, "SnowDepth_avg": 0.0,
        "WindSpeed_avg": 8.5,
    },
    "oats": {
        "opt_temp_min": 13.33, "opt_temp_max": 21.0, "opt_precip": 68.33,
        "stage_duration_days": 56.67, "temp_deviation": 2.51,
        "precip_deviation": 0.89, "water_stress_indicator": 0.673,
        "total_extreme_weather_days": 0.10, "hail_events": 0.1,
        "tornado_events": 0.01, "high_wind_events": 0.5, "thunder_events": 1.2,
        "GDD": 12.75, "precip_total": 4.41, "precip_avg": 0.15,
        "precip_max": 0.95, "Temp_avg": 62.87, "TempMAX_avg": 71.09,
        "TempMIN_avg": 54.47, "Snow_total": 0.5, "SnowDepth_avg": 0.1,
        "WindSpeed_avg": 9.2,
    },
    "soybeans": {
        "opt_temp_min": 19.0, "opt_temp_max": 26.0, "opt_precip": 91.67,
        "stage_duration_days": 83.67, "temp_deviation": 47.62,
        "precip_deviation": 85.56, "water_stress_indicator": 0.634,
        "total_extreme_weather_days": 10.30, "hail_events": 0.4,
        "tornado_events": 0.04, "high_wind_events": 1.1, "thunder_events": 2.9,
        "GDD": 34.93, "precip_total": 6.11, "precip_avg": 0.20,
        "precip_max": 1.30, "Temp_avg": 70.12, "TempMAX_avg": 80.57,
        "TempMIN_avg": 59.86, "Snow_total": 0.0, "SnowDepth_avg": 0.0,
        "WindSpeed_avg": 8.3,
    },
    "wheat": {
        "opt_temp_min": 10.0, "opt_temp_max": 24.0, "opt_precip": 65.0,
        "stage_duration_days": 93.0, "temp_deviation": 47.68,
        "precip_deviation": 94.02, "water_stress_indicator": 0.630,
        "total_extreme_weather_days": 10.54, "hail_events": 0.08,
        "tornado_events": 0.01, "high_wind_events": 0.0, "thunder_events": 10.45,
        "GDD": 34.63, "precip_total": 6.79, "precip_avg": 0.13,
        "precip_max": 1.62, "Temp_avg": 68.47, "TempMAX_avg": 78.82,
        "TempMIN_avg": 58.22, "Snow_total": 5.0, "SnowDepth_avg": 1.0,
        "WindSpeed_avg": 9.5,
    },
}

SOIL_DEFAULTS = {
    "avg_ph": 6.17,
    "avg_organic_matter_pct": 5.19,
    "avg_water_capacity": 0.190,
    "avg_bulk_density": 1.350,
    "avg_clay_pct": 20.86,
    "avg_sand_pct": 26.69,
    "avg_silt_pct": 52.01,
    "avg_cation_exchange_capacity": 17.63,
    "avg_saturated_hydraulic_conductivity": 13.71,
    "avg_soil_temperature_0_to_7cm": 12.36,
    "avg_soil_moisture_0_to_7cm": 0.317,
}

# Load all models at startup
_models = {}

for crop, filename in MODEL_FILES.items():
    if filename is None:
        continue
    path = os.path.join(BASE_DIR, "models", filename)
    if not os.path.exists(path):
        logger.warning(f"Model file not found: {path}. Yield prediction for '{crop}' will return None until the file is added.")
        continue
    try:
        with open(path, "rb") as f:
            _models[crop] = pickle.load(f)
        logger.info(f"Loaded RF model for {crop} ({_models[crop]['model'].n_estimators} trees, {len(_models[crop]['features'])} features)")
    except Exception as e:
        logger.error(f"Failed to load model for {crop}: {e}")


def predict_yield(crop: str, farm_size_hectares: float,
                  planting_date: str = None, weather_data: Optional[dict] = None) -> dict:
    """
    Predict yield for a given crop and farm.

    Return dict with:
        model_available (bool)
        yield_per_ha    (float / None) — bu/acre
        total_yield     (float / None) — total bushels for the farm
    """
    crop = crop.lower()

    if crop not in MODEL_FILES:
        return {"model_available": False, "yield_per_ha": None, "total_yield": None,
                "note": f"Unknown crop: {crop}"}

    if MODEL_FILES[crop] is None:
        return {"model_available": False, "yield_per_ha": None, "total_yield": None,
                "note": f"No model available for {crop} yet."}

    if crop not in _models:
        return {"model_available": False, "yield_per_ha": None, "total_yield": None,
                "note": f"Model file missing. Add {MODEL_FILES[crop]} to backend/models/."}

    try:
        model_dict = _models[crop]
        rf         = model_dict["model"]
        features   = model_dict["features"]

        # Build feature vector using defaults, override with live weather if provided
        defaults = {**WEATHER_DEFAULTS[crop], **SOIL_DEFAULTS}
        if weather_data:
            defaults.update(weather_data)

        feature_vector = np.array([[defaults.get(f, 0) for f in features]])
        yield_bu_acre  = float(rf.predict(feature_vector)[0])

        # Total yield: 1 ha = 2.471 acres
        acres       = farm_size_hectares * 2.471
        total_yield = round(yield_bu_acre * acres, 1)

        return {
            "model_available": True,
            "yield_per_ha":    round(yield_bu_acre, 2),
            "total_yield":     total_yield,
            "unit":            "bu/acre",
        }

    except Exception as e:
        logger.error(f"Prediction failed for {crop}: {e}")
        return {"model_available": False, "yield_per_ha": None, "total_yield": None,
                "note": str(e)}
