from typing import Optional

import os
import logging
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
from database import supabase

load_dotenv()
logger = logging.getLogger(__name__)

OWM_API_KEY  = os.environ.get("OWM_API_KEY", "")
OWM_CURRENT  = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast"


# Alert threshold constants 

IRRIGATION_THRESHOLD = {
    "maize":    25.0,   # mm / 7 days
    "soybeans": 25.0,
    "wheat":    20.0,
    "oats":     20.0,
}

HEAT_STRESS_THRESHOLD = {
    "maize":    35.0,   # °C max temp
    "soybeans": 35.0,
    "oats":     30.0,
    # wheat heat stress is handled separately (frost / Fusarium)
}

FROST_THRESHOLD = {
    "maize":    0.0,    # °C min temp
    "soybeans": 0.0,
    "wheat":    -2.0,   # during heading
    "oats":     -2.0,   # during heading
}



# OWM helpers

def _owm_get(url: str, params: dict) -> Optional[dict]:
   
    if not OWM_API_KEY or OWM_API_KEY == "owm key":
        logger.warning("OWM_API_KEY not configured. Weather fetch skipped.")
        return None
    try:
        resp = requests.get(url, params={**params, "appid": OWM_API_KEY,
                                          "units": "metric"}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"OWM request failed: {e}")
        return None


def fetch_current_weather(lat: float, lon: float) -> Optional[dict]:
   
    data = _owm_get(OWM_CURRENT, {"lat": lat, "lon": lon})
    if not data:
        return None

    return {
        "temp_avg_c":   data["main"]["temp"],
        "temp_min_c":   data["main"]["temp_min"],
        "temp_max_c":   data["main"]["temp_max"],
        "humidity_pct": data["main"]["humidity"],
        "rainfall_mm":  data.get("rain", {}).get("1h", 0.0) * 24,  # scale 1h → daily
        "description":  data["weather"][0]["description"],
        "wind_ms":      data["wind"]["speed"],
    }


def fetch_5day_forecast(lat: float, lon: float) -> list:
    
    data = _owm_get(OWM_FORECAST, {"lat": lat, "lon": lon})
    if not data:
        return []

    # Aggregate 3-hour slots → daily
    daily: dict[str, dict] = {}
    for item in data.get("list", []):
        day_str = item["dt_txt"][:10]
        t       = item["main"]["temp"]
        precip  = item.get("rain", {}).get("3h", 0.0)
        hum     = item["main"]["humidity"]
        wind    = item["wind"]["speed"]

        if day_str not in daily:
            daily[day_str] = {"tmax": t, "tmin": t, "tavg_sum": t,
                               "precip": precip, "humidity_sum": hum,
                               "wind_sum": wind, "count": 1}
        else:
            d = daily[day_str]
            d["tmax"]        = max(d["tmax"], t)
            d["tmin"]        = min(d["tmin"], t)
            d["tavg_sum"]   += t
            d["precip"]     += precip
            d["humidity_sum"] += hum
            d["wind_sum"]   += wind
            d["count"]      += 1

    result = []
    for day_str, d in sorted(daily.items()):
        from datetime import datetime
        result.append({
            "date":     datetime.strptime(day_str, "%Y-%m-%d").date(),
            "tmax":     d["tmax"],
            "tmin":     d["tmin"],
            "tavg":     d["tavg_sum"] / d["count"],
            "precip":   d["precip"],
            "humidity": d["humidity_sum"] / d["count"],
            "wind":     d["wind_sum"] / d["count"],
        })
    return result



# weather_log helpers

def log_today_weather(current: dict) -> None:
    """Insert today's weather into weather_log. Skips if already logged."""
    today = date.today().isoformat()
    existing = (supabase.table("weather_log")
                         .select("id")
                         .eq("observation_date", today)
                         .execute())
    if existing.data:
        logger.info(f"Weather already logged for {today}. Skipping insert.")
        return

    supabase.table("weather_log").insert({
        "observation_date": today,
        "temp_min_c":       current["temp_min_c"],
        "temp_max_c":       current["temp_max_c"],
        "temp_avg_c":       current["temp_avg_c"],
        "rainfall_mm":      current["rainfall_mm"],
        "humidity_pct":     current["humidity_pct"],
    }).execute()
    logger.info(f"Logged weather for {today}.")


def get_rolling_7day_rainfall() -> float:
    """Return sum of rainfall_mm for the last 7 days from weather_log."""
    since = (date.today() - timedelta(days=6)).isoformat()
    rows = (supabase.table("weather_log")
                     .select("rainfall_mm")
                     .gte("observation_date", since)
                     .execute())
    return sum(r["rainfall_mm"] or 0.0 for r in rows.data)


def get_consecutive_dry_days() -> int:
    """Count how many consecutive days ending today had rainfall < 1 mm."""
    rows = (supabase.table("weather_log")
                     .select("observation_date, rainfall_mm")
                     .order("observation_date", desc=True)
                     .limit(14)
                     .execute())
    count = 0
    for row in rows.data:
        if (row["rainfall_mm"] or 0.0) < 1.0:
            count += 1
        else:
            break
    return count


def get_consecutive_humid_days() -> int:
    """Count consecutive days (ending today) with humidity > 80 %."""
    rows = (supabase.table("weather_log")
                     .select("observation_date, humidity_pct")
                     .order("observation_date", desc=True)
                     .limit(7)
                     .execute())
    count = 0
    for row in rows.data:
        if (row["humidity_pct"] or 0.0) > 80.0:
            count += 1
        else:
            break
    return count


# Alert insertion

def _alert_exists_today(title_contains: str) -> bool:
    """Return True if an alert with this title fragment already exists today."""
    today = date.today().isoformat()
    rows = (supabase.table("tasks")
                     .select("id")
                     .eq("task_date", today)
                     .eq("is_alert", True)
                     .ilike("task_title", f"%{title_contains}%")
                     .execute())
    return len(rows.data) > 0


def _insert_alert(stage_code: str, title: str, description: str,
                  priority: str = "warning") -> None:
    """Insert one alert task for today (skip if duplicate)."""
    if _alert_exists_today(title[:30]):
        logger.info(f"Duplicate alert suppressed: {title}")
        return

    supabase.table("tasks").insert({
        "task_date":        date.today().isoformat(),
        "stage_code":       stage_code,
        "task_type":        "alert",
        "task_title":       title,
        "task_description": description,
        "is_completed":     False,
        "is_alert":         True,
        "priority":         priority,
    }).execute()
    logger.info(f"Alert inserted: {title}")



# Main daily cron function 

def run_daily_check() -> None:
    """
    Full daily cron sequence 
    1. Fetch current weather from OWM for the farm lat/lon.
    2. Insert today's record into weather_log.
    3. Query current_crop for crop type and planting date.
    4. Calculate days_from_planting.
    5. Look up current stage from growth_stages.
    6. Update current_crop.current_stage and days_from_planting.
    7. Calculate 7-day rolling rainfall.
    8. Compare to thresholds and insert alerts if needed.
    """
    # Get farm location
    farm_rows = supabase.table("demo_farm").select("*").limit(1).execute()
    if not farm_rows.data:
        logger.warning("No demo_farm record found. Skipping daily check.")
        return
    farm = farm_rows.data[0]

    # Fetch and log weather
    current = fetch_current_weather(farm["latitude"], farm["longitude"])
    if current is None:
        logger.warning("Could not fetch weather. Skipping daily check.")
        return
    log_today_weather(current)

    # Get active crop
    crop_rows = (supabase.table("current_crop")
                          .select("*")
                          .eq("status", "active")
                          .limit(1)
                          .execute())
    if not crop_rows.data:
        logger.info("No active crop. Skipping alert checks.")
        return
    crop_record = crop_rows.data[0]
    crop         = crop_record["crop"]
    planting_str = crop_record["planting_date"]

    from datetime import datetime
    planting_date = datetime.strptime(planting_str, "%Y-%m-%d").date()

    # Days from planting
    today = date.today()
    days_from_planting = (today - planting_date).days

    #  Look up current stage
    stage_rows = (supabase.table("growth_stages")
                           .select("stage_code")
                           .eq("crop", crop)
                           .lte("days_from_planting_min", days_from_planting)
                           .gte("days_from_planting_max", days_from_planting)
                           .execute())
    current_stage = (stage_rows.data[0]["stage_code"]
                     if stage_rows.data else "reproductive")

    # Update current_crop
    supabase.table("current_crop").update({
        "current_stage":     current_stage,
        "days_from_planting": days_from_planting,
    }).eq("id", crop_record["id"]).execute()
    logger.info(f"Updated current_crop: stage={current_stage}, day={days_from_planting}")

    # Threshold checks and alerts
    rolling_rain  = get_rolling_7day_rainfall()
    tmax          = current["temp_max_c"]
    tmin          = current["temp_min_c"]
    humidity      = current["humidity_pct"]
    humid_days    = get_consecutive_humid_days()
    dry_days      = get_consecutive_dry_days()

    #Irrigation alert (all crops, vegetative or reproductive)
    if current_stage in ("vegetative", "reproductive"):
        threshold = IRRIGATION_THRESHOLD.get(crop, 25.0)
        if rolling_rain < threshold:
            _insert_alert(
                current_stage,
                "Attention: Irrigation Required",
                f"7-day rolling rainfall ({rolling_rain:.1f} mm) is below the "
                f"{threshold} mm threshold for {crop} in the {current_stage} stage. "
                "Apply 25 mm of irrigation within 48 hours.",
                priority="warning",
            )

    #Heat stress for maize and soybeans (reproductive) 
    if crop in ("maize", "soybeans") and current_stage == "reproductive":
        if tmax > HEAT_STRESS_THRESHOLD[crop]:
            _insert_alert(
                current_stage,
                "🌡️ Heat Stress Alert",
                f"Maximum temperature ({tmax:.1f} °C) exceeds the {HEAT_STRESS_THRESHOLD[crop]} °C "
                f"threshold during {crop} reproductive stage. "
                "Irrigate if possible and note potential yield impact.",
                priority="critical",
            )

    #  Heat stress for oats (reproductive) 
    if crop == "oats" and current_stage == "reproductive":
        if tmax > HEAT_STRESS_THRESHOLD["oats"]:
            _insert_alert(
                current_stage,
                "🌡️ Heat Stress Alert — Oats",
                f"Maximum temperature ({tmax:.1f} °C) exceeds 30 °C during oat grain fill. "
                "Monitor grain quality closely and consider early harvest if conditions persist.",
                priority="warning",
            )

    # Frost alert (maize/soybeans in germination/early vegetative)
    if crop in ("maize", "soybeans") and current_stage == "germination":
        if tmin < FROST_THRESHOLD[crop]:
            _insert_alert(
                current_stage,
                "❄️ Frost Alert",
                f"Forecast minimum temperature ({tmin:.1f} °C) is below 0 °C "
                "during germination. Delay planting or protect emerged seedlings.",
                priority="critical",
            )

    # Frost alert (wheat/oats in reproductive/heading) 
    if crop in ("wheat", "oats") and current_stage == "reproductive":
        if tmin < FROST_THRESHOLD[crop]:
            _insert_alert(
                current_stage,
                "❄️ Spring Frost Alert",
                f"Forecast minimum temperature ({tmin:.1f} °C) is below "
                f"{FROST_THRESHOLD[crop]} °C during heading/flowering. "
                "This can cause significant sterility and yield loss.",
                priority="critical",
            )

    # Wheat: Fusarium / disease risk (reproductive) 
    if crop == "wheat" and current_stage == "reproductive":
        if 15 <= current["temp_avg_c"] <= 30 and humidity > 90:
            _insert_alert(
                current_stage,
                "🍄 Fusarium Head Blight Risk",
                f"Conditions favour Fusarium head blight: temperature {current['temp_avg_c']:.1f} °C "
                f"and humidity {humidity:.0f} %. Apply strobilurin + triazole fungicide immediately.",
                priority="critical",
            )

    # Wheat: winterkill risk (vegetative / dormancy) 
    if crop == "wheat" and current_stage == "vegetative":
        # Proxy check: if tmin is very low and it has been for several days
        # (true soil temp monitoring requires a sensor; we use air tmin as proxy)
        if tmin < -10:
            _insert_alert(
                current_stage,
                "❄️ Winterkill Risk",
                f"Air minimum temperature ({tmin:.1f} °C) is very low with no recorded "
                "snow cover data. Inspect stand in spring and assess crown rot. "
                "Replant if >50 % of plants are damaged.",
                priority="critical",
            )

    #Oats: crown rust risk (vegetative) 
    if crop == "oats" and current_stage == "vegetative":
        if humid_days >= 3:
            _insert_alert(
                current_stage,
                "🍂 Crown Rust Alert — Oats",
                f"Humidity has been above 80 % for {humid_days} consecutive days. "
                "Scout for crown rust (orange pustules on leaf undersides). "
                "Apply fungicide if moderate to high pressure is confirmed.",
                priority="warning",
            )

    #Soybeans: aphid risk (vegetative) 
    if crop == "soybeans" and current_stage == "vegetative":
        if dry_days >= 7 and tmax > 30:
            _insert_alert(
                current_stage,
                "🐛 Soybean Aphid Risk",
                f"{dry_days} consecutive dry days with high temperatures ({tmax:.1f} °C). "
                "Conditions favour soybean aphid population explosions. "
                "Scout immediately — economic threshold is 250 aphids/plant on >80 % of plants.",
                priority="warning",
            )

    logger.info("Daily cron check complete.")
