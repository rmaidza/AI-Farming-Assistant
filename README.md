# AI Farming Assistant

A full-stack web application that helps Ohio farmers make data-driven decisions about crop selection and yield forecasting. Built as a senior design project at Cleveland State University.

## Overview

The system integrates real agricultural data with machine learning and rule-based agronomic logic to deliver:

- Crop recommendations based on Ohio planting windows and today's date
- Yield predictions using trained Random Forest models (maize, soybeans, oats, wheat)
- A full-season task calendar with fertilizer, scouting, herbicide, and harvest milestones
- Automated weather alerts** for irrigation deficits, heat stress, frost risk, and disease pressure

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite |
| Backend | Python, Flask, APScheduler |
| Database | Supabase (PostgreSQL) |
| ML Models | Scikit-learn (Random Forest) |
| Weather | OpenWeatherMap API |

## Project Structure

```
farmAssistant/
├── backend/
│   ├── app.py                  # Flask API + APScheduler
│   ├── database.py             # Supabase client
│   ├── crop_recommender.py     # Rule-based crop recommendation
│   ├── calendar_generator.py   # Season task calendar generator
│   ├── weather_monitor.py      # Daily weather check + alert system
│   ├── yield_predictor.py      # Random Forest yield prediction
│   ├── models/                 # .pkl model files 
│   ├── requirements.txt
│   └── .env                    # Environment variables 
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Onboarding.jsx
│   │   │   ├── Dashboard.jsx
│   │   │   ├── TaskCard.jsx
│   │   │   └── WeatherWidget.jsx
│   │   ├── App.jsx
│   │   └── index.css
│   ├── index.html
│   └── package.json
├── database/
│   ├── schema.sql           # Table definitions
│   └── seed_growth_stages.sql
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Supabase account
- An OpenWeatherMap API key

### Backend Setup
cd backend
pip install -r requirements.txt
# Copy .env and fill in  Supabase URL, anon key, and OWM API key
python3 app.py


### Frontend Setup
cd frontend
npm install
npm run dev
```

## Team

- Russell Maidza
- Steven Vetrano
- Anesu Ruzvidzo
- Brianne Kelley

*Senior Design Project — Cleveland State University, 2026*  
*Advisor: Dr. Sunnie Chung*

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
