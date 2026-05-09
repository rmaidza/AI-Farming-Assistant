"""
calendar_generator.py
Generates all scheduled tasks for a crop season and inserts them into
the tasks table. Called once at POST /api/setup time.

"""

import logging
from datetime import date, timedelta
from database import supabase

logger = logging.getLogger(__name__)


# Task templates per crop.
# Each entry: (days_from_plant, stage_code, task_type, title, description)

TASK_TEMPLATES = {

    "maize": [
        (0,   "germination",  "planting",
         "Plant Maize",
         "Plant seeds 1.5–2 inches deep, 8–12 inches apart, 30-inch row spacing. "
         "Apply 30 lbs N/acre as starter fertilizer."),

        (7,   "germination",  "scouting",
         "Germination Check",
         "Check germination progress. Target: >90 % emergence. "
         "Scout for cutworm damage at base of seedlings."),

        (14,  "germination",  "herbicide",
         "Pre-Emergent Herbicide",
         "Apply pre-emergent herbicide (atrazine-based mix) before V3 stage. "
         "Follow label rates."),

        (30,  "vegetative",   "milestone",
         "Vegetative Stage Begins",
         "Rapid leaf development underway. Increase water and nutrient monitoring frequency."),

        (35,  "vegetative",   "fertilizer",
         "Sidedress Nitrogen Application",
         "Apply sidedress nitrogen: 100–150 lbs N/acre. "
         "Best applied at V5–V6 (knee height). Band 2 inches beside the row."),

        (50,  "vegetative",   "scouting",
         "European Corn Borer Scout",
         "Scout for European corn borer (ECB). Check leaf whorls for feeding. "
         "Apply Bt or pyrethroid if >30 % of plants are infested."),

        (65,  "vegetative",   "scouting",
         "Foliar Disease Scout",
         "Scout for gray leaf spot and northern corn leaf blight. "
         "Apply fungicide if >10 % of plants show lesions above the ear leaf."),

        (80,  "reproductive", "milestone",
         "Reproductive Stage Begins — Tasseling",
         "Protect silks. Peak water demand — irrigate aggressively if needed."),

        (85,  "reproductive", "scouting",
         "Silk & Rootworm Check",
         "Check silk development. Inspect for corn rootworm silk clipping "
         "(>1 clip per minute = threshold for action)."),

        (110, "reproductive", "scouting",
         "Stalk Rot & Harvest Timing Assessment",
         "Scout for late-season stalk rots (press stalks — should not collapse easily). "
         "Begin harvest timing assessment."),

        (130, "reproductive", "milestone",
         "Physiological Maturity Check",
         "Check for black layer at kernel base (physiological maturity). "
         "Grain should be ~30 % moisture."),

        (156, "reproductive", "harvest",
         "Harvest Maize",
         "Harvest when grain moisture reaches 15–20 %. "
         "Adjust combine settings for corn. Store at <15 % moisture."),
    ],

    "soybeans": [
        (0,   "germination",  "planting",
         "Plant Soybeans",
         "Plant seeds 1–1.5 inches deep. Inoculate with Bradyrhizobium japonicum "
         "if field has not grown soybeans in 3+ years. Apply pre-emergent herbicide."),

        (10,  "germination",  "scouting",
         "Emergence Check",
         "Check emergence. Target: 100,000–140,000 plants/acre. "
         "Scout for seedling disease (Phytophthora)."),

        (22,  "vegetative",   "milestone",
         "Vegetative Stage Begins (V1)",
         "Canopy closure begins over the next 4–6 weeks."),

        (30,  "vegetative",   "herbicide",
         "Post-Emergent Herbicide",
         "Apply post-emergent herbicide if weed pressure persists "
         "above threshold (>1 weed per foot of row)."),

        (50,  "vegetative",   "scouting",
         "Soybean Aphid Scout",
         "Scout for soybean aphid. Economic threshold: 250 aphids/plant on >80 % of plants. "
         "Check for sudden death syndrome symptoms."),

        (71,  "reproductive", "milestone",
         "Reproductive Stage Begins (R1 — First Flowers)",
         "Critical water demand period starts. Irrigate if 7-day rainfall < 25 mm."),

        (75,  "reproductive", "scouting",
         "Pod Pest Scout",
         "Scout for bean leaf beetle and Japanese beetle pod feeding. "
         "Apply insecticide if >30 % defoliation."),

        (80,  "reproductive", "fungicide",
         "White Mold Prevention Fungicide",
         "Apply fungicide for white mold prevention if canopy is dense and conditions are humid. "
         "Also apply if foliar diseases exceed 10 % severity."),

        (100, "reproductive", "scouting",
         "Stink Bug Scout",
         "Scout for stink bugs (brown and green). "
         "Economic threshold: 1 per foot of row during R3–R5."),

        (120, "reproductive", "scouting",
         "Pod Fill & Green Stem Check",
         "Pods should be full (R6). Check for green stem syndrome (maturation delay in wet years). "
         "Scout for pod shattering risk."),

        (151, "reproductive", "harvest",
         "Harvest Soybeans",
         "Harvest at 13–15 % moisture. Pods should rattle when shaken. "
         "Lower header to minimise shattering losses. Store at <13 % moisture."),
    ],

    "wheat": [
        (0,   "germination",  "planting",
         "Plant Winter Wheat",
         "Sow at 1.5–2 inches depth, 6–7 inch row spacing. "
         "Apply 30–40 lbs N/acre starter + full P and K at seeding. "
         "Check seed treatment for fungicide/insecticide coating."),

        (14,  "germination",  "scouting",
         "Stand Density Check",
         "Check emergence and stand density. Target: 25–30 plants per foot of row. "
         "Scout for Hessian fly."),

        (30,  "germination",  "herbicide",
         "Fall Herbicide Application",
         "Apply fall herbicide for broadleaf weeds (2,4-D amine or dicamba blend). "
         "Apply before canopy closure."),

        (38,  "vegetative",   "milestone",
         "Vegetative Stage — Overwintering Begins",
         "Record final tiller count per plant (target: 4–6). Plant enters dormancy."),

        (140, "vegetative",   "scouting",
         "Spring Greenup Check",
         "Assess winter survival (~mid-March). Dig plants and check for crown rot. "
         "Determine if replanting is needed."),

        (172, "vegetative",   "fertilizer",
         "Spring Topdress Nitrogen",
         "Apply 60–80 lbs N/acre at jointing (Feekes 6). "
         "Do not delay — apply within 1 week of jointing detection."),

        (173, "reproductive", "milestone",
         "Reproductive Stage Begins — Jointing",
         "Flag leaf emerging. Fungicide timing window opens."),

        (185, "reproductive", "fungicide",
         "Flag Leaf Fungicide Application",
         "Apply fungicide at Feekes 8–9 (flag leaf fully emerged): "
         "strobilurin + triazole mix for stripe rust, powdery mildew, and Septoria control."),

        (200, "reproductive", "fungicide",
         "Heading Fungicide — FHB Prevention",
         "At 50 % heads emerged (BBCH 55): apply fungicide specifically for "
         "Fusarium head blight (FHB) if weather is humid. Use triazole fungicide."),

        (215, "reproductive", "scouting",
         "Aphid & Late Disease Scout",
         "Flowering complete. Monitor for aphids in grain fill. "
         "Scout for late-season diseases."),

        (250, "reproductive", "scouting",
         "Harvest Readiness Assessment",
         "Assess harvest readiness. Check grain moisture with a hand meter. "
         "Target: 13–14 % for direct storage."),

        (279, "reproductive", "harvest",
         "Harvest Winter Wheat",
         "Harvest at 13–14 % moisture. Adjust combine for small grain. "
         "Avoid harvesting above 14 % for storage. Check for mycotoxin risk if wet spring."),
    ],

    "oats": [
        (0,   "germination",  "planting",
         "Plant Oats",
         "Sow 1–1.5 inches deep at 2–3 bu/acre. "
         "Apply 40–50 lbs P2O5 and 60–80 lbs K2O at seeding. "
         "No starter N needed if fall N was applied."),

        (10,  "germination",  "scouting",
         "Stand Check & Aphid Scout",
         "Check stand (target: 20–25 plants per foot of row). "
         "Scout for aphids transmitting Barley Yellow Dwarf Virus."),

        (21,  "vegetative",   "milestone",
         "Vegetative Stage Begins",
         "Rapid tillering phase underway."),

        (25,  "vegetative",   "herbicide",
         "Herbicide Application",
         "Apply pre-emergent or post-emergent herbicide for wild oats and broadleaf weeds."),

        (40,  "vegetative",   "fertilizer",
         "Topdress Nitrogen Application",
         "Apply topdress N: 40–60 lbs N/acre at jointing. "
         "Avoid excess N — increases lodging risk."),

        (50,  "vegetative",   "scouting",
         "Crown Rust Scout",
         "Scout for crown rust (orange pustules on leaf undersides). "
         "Apply fungicide if moderate to high pressure."),

        (71,  "reproductive", "milestone",
         "Reproductive Stage Begins — Heading",
         "Oats are self-pollinating. Continue rust scouting."),

        (85,  "reproductive", "scouting",
         "Grain Fill & Ergot Check",
         "Check grain fill progress. Scout for ergot (black spurs replacing kernels) "
         "if conditions are wet."),

        (110, "reproductive", "scouting",
         "Harvest Readiness Assessment",
         "Assess grain moisture. Plan harvest logistics. "
         "Oats lose quality quickly if rained on at maturity."),

        (131, "reproductive", "harvest",
         "Harvest Oats",
         "Harvest at <14 % moisture. Reduce cylinder/rotor speed to minimise kernel cracking. "
         "Store at <13 % moisture."),
    ],
}



# Stage code resolver

def _resolve_stage(crop: str, days_from_plant: int) -> str:
    """Return the stage_code that contains the given day offset."""
    STAGE_RANGES = {
        "maize":    [("germination", 0, 30),   ("vegetative", 31, 80),   ("reproductive", 81, 156)],
        "soybeans": [("germination", 0, 22),   ("vegetative", 23, 70),   ("reproductive", 71, 151)],
        "wheat":    [("germination", 0, 37),   ("vegetative", 38, 172),  ("reproductive", 173, 280)],
        "oats":     [("germination", 0, 21),   ("vegetative", 22, 71),   ("reproductive", 72, 131)],
    }
    for stage_code, s_min, s_max in STAGE_RANGES.get(crop, []):
        if s_min <= days_from_plant <= s_max:
            return stage_code
    return "reproductive"  # fallback for any day beyond the last stage



# Public API

def generate_calendar(crop: str, planting_date: date) -> int:
    """
    Generate and insert all scheduled tasks for the given crop and planting date.
    Returns the number of tasks inserted.

    Existing tasks are deleted first (full reset on new setup, safe for demo).
    """
    if crop not in TASK_TEMPLATES:
        raise ValueError(f"Unknown crop: {crop}. Must be one of {list(TASK_TEMPLATES.keys())}")

    # Clear previous tasks (demo resets overwrite everything)
    supabase.table("tasks").delete().neq("id", 0).execute()
    logger.info("Cleared existing tasks table.")

    templates = TASK_TEMPLATES[crop]
    rows = []

    for (days_offset, stage_code, task_type, title, description) in templates:
        task_date = planting_date + timedelta(days=days_offset)
        rows.append({
            "task_date":        task_date.isoformat(),
            "stage_code":       stage_code,
            "task_type":        task_type,
            "task_title":       title,
            "task_description": description,
            "is_completed":     False,
            "completed_date":   None,
            "is_alert":         False,
            "priority":         "normal",
        })

    if rows:
        supabase.table("tasks").insert(rows).execute()
        logger.info(f"Inserted {len(rows)} tasks for {crop} planted on {planting_date}.")

    return len(rows)
