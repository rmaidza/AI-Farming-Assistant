-- AI FARMING ASSISTANT  DATABASE SCHEMA

-- Enable UUID generation 
CREATE EXTENSION IF NOT EXISTS "pgcrypto";



-- TABLE 1: demo_farm
-- Holds exactly one record representing the demo farm.
-- Reset between sessions by truncating and re-seeding.

CREATE TABLE IF NOT EXISTS demo_farm (
    id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    farm_name           VARCHAR(200)    NOT NULL DEFAULT 'Demo Farm',
    location            VARCHAR(200)    NOT NULL,
    latitude            DECIMAL(10,7)   NOT NULL,
    longitude           DECIMAL(10,7)   NOT NULL,
    farm_size_hectares  DECIMAL(10,2)   NOT NULL
);



-- TABLE 2: current_crop
-- One active crop planting per demo session.
-- Updated or replaced when farmer resets / starts over.

CREATE TABLE IF NOT EXISTS current_crop (
    id                       UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    crop                     VARCHAR(50)     NOT NULL
                                             CHECK (crop IN ('maize', 'soybeans', 'wheat', 'oats')),
    planting_date            DATE            NOT NULL,
    expected_harvest_date    DATE            NOT NULL,
    predicted_yield_per_ha   DECIMAL(10,2)   NULL,           -- ML model output (bu/acre per ha basis)
    current_stage            VARCHAR(20)     NOT NULL
                                             CHECK (current_stage IN ('germination', 'vegetative', 'reproductive')),
    days_from_planting       INTEGER         NOT NULL DEFAULT 0,
    status                   VARCHAR(20)     NOT NULL DEFAULT 'active'
                                             CHECK (status IN ('active', 'complete', 'abandoned'))
);



-- TABLE 3: growth_stages
-- Pre-populated reference table.
-- 3 stages × 4 crops = 12 rows total.


CREATE TABLE IF NOT EXISTS growth_stages (
    id                      SERIAL          PRIMARY KEY,
    crop                    VARCHAR(50)     NOT NULL
                                            CHECK (crop IN ('maize', 'soybeans', 'wheat', 'oats')),
    stage_code              VARCHAR(20)     NOT NULL
                                            CHECK (stage_code IN ('germination', 'vegetative', 'reproductive')),
    stage_name              VARCHAR(100)    NOT NULL,
    days_from_planting_min  INTEGER         NOT NULL,
    days_from_planting_max  INTEGER         NOT NULL,
    description             TEXT,
    temp_min_c              DECIMAL(5,2)    NOT NULL,
    temp_max_c              DECIMAL(5,2)    NOT NULL,
    precip_min_mm           DECIMAL(7,2)    NOT NULL,
    precip_max_mm           DECIMAL(7,2)    NOT NULL,

    CONSTRAINT uq_crop_stage UNIQUE (crop, stage_code),
    CONSTRAINT chk_days_order CHECK (days_from_planting_max >= days_from_planting_min)
);


-- TABLE 4: weather_log
-- One row per day. 
-- Used to calculate 7-day rolling rainfall, GDD, and alert thresholds.

CREATE TABLE IF NOT EXISTS weather_log (
    id                SERIAL          PRIMARY KEY,
    observation_date  DATE            NOT NULL UNIQUE,
    temp_min_c        DECIMAL(5,2),
    temp_max_c        DECIMAL(5,2),
    temp_avg_c        DECIMAL(5,2),
    rainfall_mm       DECIMAL(7,2),
    humidity_pct      DECIMAL(5,2)
);



-- TABLE 5: tasks
-- Generated at setup time (scheduled tasks) and appended dynamically by weather_monitor.py (alert tasks).

CREATE TABLE IF NOT EXISTS tasks (
    id                SERIAL          PRIMARY KEY,
    task_date         DATE            NOT NULL,
    stage_code        VARCHAR(20)
                                      CHECK (stage_code IN ('germination', 'vegetative', 'reproductive')),
    task_type         VARCHAR(50)
                                      CHECK (task_type IN ('planting', 'fertilizer', 'irrigation',
                                                           'scouting', 'harvest', 'alert',
                                                           'milestone', 'herbicide', 'fungicide')),
    task_title        VARCHAR(200)    NOT NULL,
    task_description  TEXT,
    is_completed      BOOLEAN         NOT NULL DEFAULT FALSE,
    completed_date    DATE            NULL,
    is_alert          BOOLEAN         NOT NULL DEFAULT FALSE,
    priority          VARCHAR(20)     NOT NULL DEFAULT 'normal'
                                      CHECK (priority IN ('normal', 'warning', 'critical'))
);

-- Index to speed up dashboard queries (upcoming tasks by date)
CREATE INDEX IF NOT EXISTS idx_tasks_task_date       ON tasks (task_date);
CREATE INDEX IF NOT EXISTS idx_tasks_is_completed    ON tasks (is_completed);
CREATE INDEX IF NOT EXISTS idx_tasks_is_alert        ON tasks (is_alert);

-- Index to speed up cron lookups on weather_log
CREATE INDEX IF NOT EXISTS idx_weather_log_date      ON weather_log (observation_date);
