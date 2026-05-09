# Mian app logic and runs daily weather CRON

from typing import Optional

import os
import logging
from datetime import date, datetime, timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv

from database import supabase
from crop_recommender import recommend_crops
from yield_predictor import predict_yield
from calendar_generator import generate_calendar
from weather_monitor import (
    run_daily_check,
    fetch_current_weather,
    fetch_5day_forecast,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # Allow React frontend on a different port during development



# APScheduler — daily cron at 06:00 (Build Guide §7.1)

scheduler = BackgroundScheduler(timezone="America/New_York")
scheduler.add_job(run_daily_check, "cron", hour=6, minute=0, id="daily_weather_check")
scheduler.start()
logger.info("APScheduler started — daily weather check at 06:00 ET.")



# Helper: get the single active current_crop record

def _get_active_crop() -> Optional[dict]:
    rows = (supabase.table("current_crop")
                     .select("*")
                     .eq("status", "active")
                     .limit(1)
                     .execute())
    return rows.data[0] if rows.data else None



# POST /api/setup
# Initialize (or reset) the demo farm, run yield prediction, generate calendar.

@app.route("/api/setup", methods=["POST"])
def setup():
    """
    Expected JSON body:
    {
        "farm_name":          "Demo Farm",        (optional)
        "location":           "Columbus, Ohio",
        "latitude":           39.9612,
        "longitude":          -82.9988,
        "farm_size_hectares": 5.0,
        "crop":               "maize",
        "planting_date":      "2025-05-01"
    }
    """
    body = request.get_json(force=True)
    required = ["location", "latitude", "longitude", "farm_size_hectares",
                "crop", "planting_date"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    crop          = body["crop"].lower()
    valid_crops   = ("maize", "soybeans", "wheat", "oats")
    if crop not in valid_crops:
        return jsonify({"error": f"crop must be one of {valid_crops}"}), 400

    try:
        planting_date = datetime.strptime(body["planting_date"], "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "planting_date must be YYYY-MM-DD"}), 400

    # Reset and seed demo_farm (one record only) 
    supabase.table("demo_farm").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    farm_row = supabase.table("demo_farm").insert({
        "farm_name":          body.get("farm_name", "Demo Farm"),
        "location":           body["location"],
        "latitude":           body["latitude"],
        "longitude":          body["longitude"],
        "farm_size_hectares": body["farm_size_hectares"],
    }).execute()
    farm = farm_row.data[0]

    # Calculate season dates 
    SEASON_DURATIONS = {
        "maize": 156, "soybeans": 151, "wheat": 280, "oats": 131,
    }
    harvest_date   = planting_date + timedelta(days=SEASON_DURATIONS[crop])
    days_planted   = (date.today() - planting_date).days
    days_planted   = max(0, days_planted)

    # Determine initial stage 
    stage_rows = (supabase.table("growth_stages")
                           .select("stage_code")
                           .eq("crop", crop)
                           .lte("days_from_planting_min", days_planted)
                           .gte("days_from_planting_max", days_planted)
                           .execute())
    initial_stage = stage_rows.data[0]["stage_code"] if stage_rows.data else "germination"

    # Fetch forecast for yield prediction 
    forecast = fetch_5day_forecast(body["latitude"], body["longitude"])

    #  Yield prediction 
    yield_result = predict_yield(
        crop=crop,
        planting_date=planting_date.isoformat(),
        weather_data=None,
        farm_size_hectares=body["farm_size_hectares"],
    )

    # Reset and seed current_crop 
    supabase.table("current_crop").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    crop_row = supabase.table("current_crop").insert({
        "crop":                   crop,
        "planting_date":          planting_date.isoformat(),
        "expected_harvest_date":  harvest_date.isoformat(),
        "predicted_yield_per_ha": yield_result["yield_per_ha"],
        "current_stage":          initial_stage,
        "days_from_planting":     days_planted,
        "status":                 "active",
    }).execute()

    # Generate task calendar 
    task_count = generate_calendar(crop, planting_date)

    return jsonify({
        "farm":                farm,
        "crop":                crop_row.data[0],
        "predicted_yield":     yield_result,
        "tasks_generated":     task_count,
        "harvest_date":        harvest_date.isoformat(),
    }), 201



# GET /api/dashboard
# Return current stage, upcoming tasks (today + 14 days), active alerts.

@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    crop_record = _get_active_crop()
    if not crop_record:
        return jsonify({"error": "No active crop. Run POST /api/setup first."}), 404

    today     = date.today()

    # Upcoming tasks (next 3 incomplete tasks from today onwards)
    task_rows = (supabase.table("tasks")
                          .select("*")
                          .gte("task_date", today.isoformat())
                          .eq("is_completed", False)
                          .eq("is_alert", False)
                          .order("task_date")
                          .limit(3)
                          .execute())

    # Active alerts (any date, not yet completed)
    alert_rows = (supabase.table("tasks")
                           .select("*")
                           .eq("is_alert", True)
                           .eq("is_completed", False)
                           .order("task_date")
                           .execute())

    # Completed tasks (for collapsible history view)
    completed_rows = (supabase.table("tasks")
                               .select("*")
                               .eq("is_completed", True)
                               .order("completed_date", desc=True)
                               .limit(50)
                               .execute())

    # Growth stage info for the current crop/stage
    stage_info = (supabase.table("growth_stages")
                           .select("stage_name, description, temp_min_c, temp_max_c")
                           .eq("crop", crop_record["crop"])
                           .eq("stage_code", crop_record["current_stage"])
                           .execute())

    return jsonify({
        "current_stage":        crop_record["current_stage"],
        "crop":                 crop_record["crop"],
        "days_from_planting":   crop_record["days_from_planting"],
        "predicted_yield_per_ha": crop_record["predicted_yield_per_ha"],
        "expected_harvest_date": crop_record["expected_harvest_date"],
        "stage_info":           stage_info.data[0] if stage_info.data else None,
        "tasks":                task_rows.data,
        "alerts":               alert_rows.data,
        "completed_tasks":      completed_rows.data,
    }), 200



# POST /api/tasks/<id>/complete
# Mark a task as completed.

@app.route("/api/tasks/<int:task_id>/complete", methods=["POST"])
def complete_task(task_id: int):
    # Verify task exists
    existing = (supabase.table("tasks")
                         .select("id, is_completed")
                         .eq("id", task_id)
                         .execute())
    if not existing.data:
        return jsonify({"error": f"Task {task_id} not found."}), 404

    task = existing.data[0]
    if task["is_completed"]:
        return jsonify({"message": "Task already marked complete.", "task": task}), 200

    updated = (supabase.table("tasks")
                        .update({
                            "is_completed":   True,
                            "completed_date": date.today().isoformat(),
                        })
                        .eq("id", task_id)
                        .execute())

    return jsonify({"message": "Task marked complete.", "task": updated.data[0]}), 200



# GET /api/weather
# Return current conditions and 5-day forecast for the demo farm.

@app.route("/api/weather", methods=["GET"])
def weather():
    farm_rows = supabase.table("demo_farm").select("*").limit(1).execute()
    if not farm_rows.data:
        return jsonify({"error": "No farm configured. Run POST /api/setup first."}), 404

    farm    = farm_rows.data[0]
    lat     = farm["latitude"]
    lon     = farm["longitude"]

    current  = fetch_current_weather(lat, lon)
    forecast = fetch_5day_forecast(lat, lon)

    # Pull last 7 days of logged rainfall for context
    since = (date.today() - timedelta(days=6)).isoformat()
    history = (supabase.table("weather_log")
                        .select("observation_date, rainfall_mm, temp_avg_c, humidity_pct")
                        .gte("observation_date", since)
                        .order("observation_date")
                        .execute())

    rolling_7d_rain = sum(r["rainfall_mm"] or 0.0 for r in history.data)

    return jsonify({
        "location":           farm["location"],
        "current":            current,
        "forecast":           [
            {**f, "date": f["date"].isoformat()} for f in forecast
        ],
        "rolling_7d_rain_mm": round(rolling_7d_rain, 2),
        "weather_history":    history.data,
    }), 200



# GET /api/recommend
# Returns recommended crops based on today's date (used by Onboarding screen).

@app.route("/api/recommend", methods=["GET"])
def recommend():
    today = date.today()
    result = recommend_crops(today.month, today.day, today.year)
    return jsonify(result), 200



# GET /api/tasks/all
# Returns every task for the active season, used by the calendar and analytics view which need the full season, not just the next 3 tasks.

@app.route("/api/tasks/all", methods=["GET"])
def all_tasks():
    crop_record = _get_active_crop()
    if not crop_record:
        return jsonify({"error": "No active crop. Run POST /api/setup first."}), 404

    rows = (supabase.table("tasks")
                    .select("*")
                    .order("task_date")
                    .execute())

    return jsonify({"tasks": rows.data}), 200



# Run

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=True, port=port, use_reloader=False)

    # side note: use_reloader=False because we found it prevents APScheduler from running twice in debug mode
