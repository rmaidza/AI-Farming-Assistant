from datetime import date, timedelta



# Total season durations in days for calculating the estimated harvest date shown to the farmer.

SEASON_DURATIONS = {
    "maize":    156,   # germination(30) + vegetative(50) + reproductive(76)
    "soybeans": 151,   # germination(22) + vegetative(48) + reproductive(81)
    "wheat":    280,   # germination(37) + vegetative(135) + reproductive(108)
    "oats":     131,   # germination(21) + vegetative(50) + reproductive(60)
}

# Ideal planting date ranges per crop for Ohio
IDEAL_WINDOWS = {
    "maize":    {"start": (4, 25), "end": (6, 30)},
    "soybeans": {"start": (5, 10), "end": (6, 30)},
    "wheat":    {"start": (9, 15), "end": (10, 15)},
    "oats":     {"start": (3, 15), "end": (5, 9)},
}


def _ideal_planting_date(crop: str, today: date) -> date:
    """
    Return the ideal planting date for a crop given today's date.
    If today falls inside the window, return today.
    If before the window, return the window open date.
    """
    month, day = IDEAL_WINDOWS[crop]["start"]
    year = today.year
    window_start = date(year, month, day)

    # Wheat window spans late Sep/early Oct 
    # Oats window is entirely in spring
    if today >= window_start:
        return today
    return window_start


def recommend_crops(current_month: int, current_day: int, year: int = None) -> dict:
    """
    Return recommended crops and planting guidance for the given date.

    Return a dict with keys:
        recommended_crops (list[str])  — crop names ready to plant
        message           (str)        — human-readable guidance
        crop_details      (list[dict]) — per-crop planting + harvest info
    """
    if year is None:
        year = date.today().year

    today = date(year, current_month, current_day)
    m, d = current_month, current_day


    if (m == 3 and d >= 15) or (m == 3 and d == 31):
        crops = ["oats"]
        message = (
            "Oats only. Cool-season crop tolerates frost — soil is workable. "
            "Do not plant maize or soybeans yet; soil temperature is too low."
        )
    elif m == 4 and d <= 24:
        crops = ["oats"]
        message = (
            "Oats only. Soil temperature is still too low for maize or soybeans. "
            "Oats remain viable through late April."
        )
    elif (m == 4 and d >= 25) or (m == 5 and d <= 9):
        crops = ["maize", "oats"]
        message = (
            "Maize planting window is now open. Oats are still viable. "
            "Verify soil temperature at 2-inch depth is ≥50 °F before planting maize."
        )
    elif (m == 5 and d >= 10) or m == 6:
        crops = ["maize", "soybeans"]
        message = (
            "Prime window for maize and soybeans. "
            "Note: late maize plantings (after June 1) reduce yield by ~1 bu/day."
        )
    elif m in (7, 8):
        crops = []
        message = (
            "No crops recommended. Too late for spring crops; "
            "too early for fall wheat planting. Season has passed for spring crops."
        )
    elif (m == 9 and d >= 15) or (m == 10 and d <= 15):
        crops = ["wheat"]
        message = (
            "Winter wheat planting window is open. "
            "Must plant before October 15 to allow adequate tillering before first frost."
        )
    elif (m == 10 and d >= 16) or m == 11:
        crops = []
        message = (
            "Too late for wheat establishment. "
            "Planting now risks winterkill due to insufficient tiller development before frost."
        )
    else:
        # Dec, Jan, Feb: no window open
        crops = []
        message = "No crops recommended for this time of year in Ohio."

    # Build per-crop detail objects
    crop_details = []
    for crop in crops:
        planting = _ideal_planting_date(crop, today)
        harvest = planting + timedelta(days=SEASON_DURATIONS[crop])
        w_start = IDEAL_WINDOWS[crop]["start"]
        w_end   = IDEAL_WINDOWS[crop]["end"]
        crop_details.append({
            "crop":                  crop,
            "ideal_planting_date":   planting.isoformat(),
            "estimated_harvest_date": harvest.isoformat(),
            "planting_window_start": date(year, w_start[0], w_start[1]).isoformat(),
            "planting_window_end":   date(year, w_end[0],   w_end[1]).isoformat(),
            "season_duration_days":  SEASON_DURATIONS[crop],
        })

    return {
        "recommended_crops": crops,
        "message":           message,
        "crop_details":      crop_details,
    }
