-- AI FARMING ASSISTANT — SEED DATA: growth_stages
-- Populates all 12 rows (3 stages × 4 crops).

--   precip_min_mm = 80% of the optimal total listed in crop_profiles.csv
--                   (lower bound — triggers irrigation alert below this)
--   precip_max_mm = the optimal total from crop_profiles.csv
--                   (upper bound — waterlogging risk above this)

-- Clear existing data 
TRUNCATE TABLE growth_stages RESTART IDENTITY CASCADE;



-- MAIZE 
-- Total season: ~156 days  |  Ohio planting window: late Apr – mid May


INSERT INTO growth_stages
    (crop, stage_code, stage_name,
     days_from_planting_min, days_from_planting_max,
     description,
     temp_min_c, temp_max_c,
     precip_min_mm, precip_max_mm)
VALUES
(
    'maize', 'germination', 'Germination & Emergence',
    0, 30,
    'Seeds absorb water and germinate. Radicle and coleoptile emerge. '
    'Soil temperature at seed depth must be ≥10 °C (50 °F). '
    'Stand establishment target: 32,000 plants/acre. '
    'Apply starter fertilizer (30 lbs N/acre) at planting. '
    'Scout for cutworms and seedling disease. '
    'Apply pre-emergent herbicide before V3.',
    10.00, 15.00,
    32.00, 40.00          -- 80 % of 40 mm optimal = 32 mm lower bound
),
(
    'maize', 'vegetative', 'Vegetative (Leaf & Stem Development)',
    31, 80,
    'Rapid leaf and stem elongation from V1 through V12. '
    'Highest nutrient demand period. Apply sidedress N (100–150 lbs/acre) by V6. '
    'Water stress during V6–V8 can reduce yield by 5–8 %. '
    'Scout for European corn borer and gray leaf spot. '
    'Irrigate if 7-day rolling rainfall < 25 mm.',
    18.00, 30.00,
    100.00, 125.00        -- 80 % of 125 mm = 100 mm lower bound
),
(
    'maize', 'reproductive', 'Reproductive (Tasseling through Maturity)',
    81, 156,
    'Tasseling (VT), silking (R1), grain fill (R2–R5), and physiological maturity (R6). '
    'Peak water demand. Temperatures >35 °C during tasseling/silking damage pollen and reduce yield. '
    'Irrigate aggressively if 7-day rainfall < 25 mm. '
    'Scout for corn rootworm silk clipping, stalk rots, and aflatoxin risk. '
    'Harvest when grain moisture is 15–20 %. Check for black layer at kernel base.',
    21.00, 30.00,
    140.00, 175.00        -- 80 % of 175 mm = 140 mm lower bound
);


-- SOYBEANS
-- Total season: ~151 days  |  Ohio planting window: early–mid May


INSERT INTO growth_stages
    (crop, stage_code, stage_name,
     days_from_planting_min, days_from_planting_max,
     description,
     temp_min_c, temp_max_c,
     precip_min_mm, precip_max_mm)
VALUES
(
    'soybeans', 'germination', 'Germination & Emergence',
    0, 22,
    'Seed imbibition and hypocotyl emergence. '
    'Soil temperature must be ≥10 °C (50 °F) at seed depth. '
    'Plant 1–1.5 inches deep; inoculate with Bradyrhizobium japonicum. '
    'Apply pre-emergent herbicide at planting. '
    'Stand target: 100,000–140,000 plants/acre. '
    'Scout for seedling disease (Phytophthora).',
    15.00, 20.00,
    40.00, 50.00          -- 80 % of 50 mm = 40 mm lower bound
),
(
    'soybeans', 'vegetative', 'Vegetative (V1 – R0)',
    23, 70,
    'Rapid leaf and canopy development from V1 through pre-flowering. '
    'Legume fixes its own nitrogen — no N fertilizer required. '
    'Apply post-emergent herbicide if weed pressure persists. '
    'Scout for soybean aphid (threshold: 250 aphids/plant on >80 % of plants) '
    'and sudden death syndrome. '
    'Aphid risk elevated during 7+ dry days with temperatures >30 °C.',
    20.00, 30.00,
    80.00, 100.00         -- 80 % of 100 mm = 80 mm lower bound
),
(
    'soybeans', 'reproductive', 'Reproductive (R1 – R8)',
    71, 151,
    'Flowering (R1–R2), pod set (R3–R4), seed fill (R5–R6), and maturity (R7–R8). '
    'Highest water demand — water stress during R1–R2 significantly reduces pod count. '
    'Irrigate if 7-day rainfall < 25 mm. '
    'Scout for bean leaf beetle, stink bugs, and white mold. '
    'Heat stress (>35 °C) during flowering reduces pod set. '
    'Harvest at 13–15 % moisture; pods should rattle when shaken.',
    22.00, 28.00,
    100.00, 125.00        -- 80 % of 125 mm = 100 mm lower bound
);



-- WINTER WHEAT
-- Total season: ~280 days (Sep–Jun, crosses calendar year)
-- Ohio planting window: late Sep – early Oct
-- NOTE: Vegetative stage covers winter dormancy (Nov–Mar).
--       Temp range reflects the dormancy window, not optimal growth temps.


INSERT INTO growth_stages
    (crop, stage_code, stage_name,
     days_from_planting_min, days_from_planting_max,
     description,
     temp_min_c, temp_max_c,
     precip_min_mm, precip_max_mm)
VALUES
(
    'wheat', 'germination', 'Germination & Fall Establishment',
    0, 37,
    'Seed germination and fall tillering before dormancy onset. '
    'Sow 1.5–2 inches deep at 6–7 inch row spacing. '
    'Apply starter N (30–40 lbs/acre) to promote tillering. '
    'Target stand: 4–6 tillers per plant by winter. '
    'Scout for Hessian fly and aphids. '
    'Apply fall herbicide for broadleaf weed control. Soil pH target: 6.0–7.0.',
    10.00, 24.00,
    32.00, 40.00          -- 80 % of 40 mm = 32 mm lower bound
),
(
    'wheat', 'vegetative', 'Vegetative & Overwintering (Dormancy)',
    38, 172,
    'Plant enters dormancy after fall tillering. Vernalization occurs during deep dormancy. '
    'Do NOT apply fertilizer during deep dormancy. '
    'Winterkill alert: soil temp < -10 °C sustained for 3+ days without snow cover. '
    'Spring greenup check (~day 140, approx. mid-March): assess winter survival, '
    'dig plants and check for crown rot, determine if replanting is needed. '
    'Apply spring topdress N (60–80 lbs/acre) at jointing (Feekes 6, ~day 172). '
    'Temp range covers full dormancy window (-7 °C min to 10 °C max).',
    -7.00, 10.00,
    52.00, 65.00          -- 80 % of 65 mm = 52 mm lower bound
),
(
    'wheat', 'reproductive', 'Reproductive (Jointing through Harvest)',
    173, 280,
    'Jointing (Feekes 6), booting, heading, flowering, grain fill, and harvest. '
    'Most critical fungicide window: Feekes 8–9 (flag leaf) for stripe rust and powdery mildew. '
    'Apply second fungicide at 50 % heading (BBCH 55) for Fusarium head blight if humid. '
    'Fusarium alert: temp 15–30 °C + humidity >90 % during heading. '
    'Spring frost alert: forecast temp < -2 °C during heading or flowering. '
    'Harvest at 13–14 % moisture; check for mycotoxin risk in wet springs.',
    15.00, 22.00,
    140.00, 175.00        -- 80 % of 175 mm = 140 mm lower bound
);


-- OATS (SPRING OATS)
-- Total season: ~131 days  |  Ohio planting window: mid–late March


INSERT INTO growth_stages
    (crop, stage_code, stage_name,
     days_from_planting_min, days_from_planting_max,
     description,
     temp_min_c, temp_max_c,
     precip_min_mm, precip_max_mm)
VALUES
(
    'oats', 'germination', 'Germination & Emergence',
    0, 21,
    'Seed germination and early emergence. '
    'Oats tolerate frost down to -2 °C — can be sown as soon as soil is workable. '
    'Plant 1–1.5 inches deep at 2–3 bu/acre seeding rate. '
    'Apply 40–50 lbs P2O5/acre and 60–80 lbs K2O/acre at seeding. '
    'Soil pH target: 5.5–7.0. '
    'Check stand: target 20–25 plants per foot of row.',
    5.00, 15.00,
    32.00, 40.00          -- 80 % of 40 mm = 32 mm lower bound
),
(
    'oats', 'vegetative', 'Vegetative (Tillering & Stem Elongation)',
    22, 71,
    'Rapid tillering followed by stem elongation (jointing). '
    'Cool, moist conditions favour this stage. '
    'Apply post-emergent herbicide for wild oat and broadleaf weed control (~day 25). '
    'Apply topdress N (40–60 lbs/acre) at jointing (~day 40). '
    'Avoid excess N — increases lodging risk. '
    'Scout for crown rust (orange pustules on leaf undersides) and aphids. '
    'Crown rust alert: humidity >80 % sustained for 3+ days — recommend fungicide.',
    12.00, 20.00,
    64.00, 80.00          -- 80 % of 80 mm = 64 mm lower bound
),
(
    'oats', 'reproductive', 'Reproductive (Heading through Harvest)',
    72, 131,
    'Heading, flowering (self-pollinating), grain fill, and maturity. '
    'Oats do not tolerate summer heat — heat stress alert if temp >30 °C during this stage. '
    'Scout for Barley Yellow Dwarf Virus (BYDV) transmitted by aphids. '
    'Scout for ergot (black spurs replacing kernels) in wet conditions. '
    'Irrigation alert: 7-day rainfall < 20 mm. '
    'Oats lose quality rapidly if rained on at maturity. '
    'Harvest at <14 % moisture; store at <13 % moisture.',
    15.00, 22.00,
    80.00, 100.00         -- 80 % of 100 mm = 80 mm lower bound
);



-- should return 12 rows

SELECT crop, stage_code, stage_name,
       days_from_planting_min, days_from_planting_max,
       temp_min_c, temp_max_c,
       precip_min_mm, precip_max_mm
FROM growth_stages
ORDER BY
    CASE crop
        WHEN 'maize'    THEN 1
        WHEN 'soybeans' THEN 2
        WHEN 'wheat'    THEN 3
        WHEN 'oats'     THEN 4
    END,
    days_from_planting_min;
