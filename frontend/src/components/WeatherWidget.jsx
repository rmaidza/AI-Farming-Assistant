import { useState, useEffect } from 'react'
import './WeatherWidget.css'

const WX_ICONS = {
  clear:       '☀️',
  clouds:      '⛅',
  rain:        '🌧️',
  drizzle:     '🌦️',
  thunderstorm:'⛈️',
  snow:        '❄️',
  mist:        '🌫️',
  default:     '🌡️',
}

function wxIcon(desc = '') {
  const d = desc.toLowerCase()
  for (const [key, icon] of Object.entries(WX_ICONS)) {
    if (d.includes(key)) return icon
  }
  return WX_ICONS.default
}

function fmt(date) {
  return new Date(date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

export default function WeatherWidget({ tempUnit = 'C' }) {
  const toTemp = (c) => tempUnit === 'F' ? (c * 9/5 + 32) : c
  const unitLabel = `°${tempUnit}`
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  useEffect(() => {
    fetch('/api/weather')
      .then(r => {
        if (!r.ok) throw new Error('Weather unavailable')
        return r.json()
      })
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="weather-widget">
      <div className="weather-header"><span className="section-icon">🌤</span> Weather</div>
      <div className="weather-skeleton">
        {[1,2,3].map(i => <div key={i} className="skeleton" style={{height:60}} />)}
      </div>
    </div>
  )

  if (error) return (
    <div className="weather-widget weather-widget--error">
      <div className="weather-header"><span className="section-icon">🌤</span> Weather</div>
      <div className="weather-unavailable">
        <span>🔌</span>
        <span>Weather data unavailable — add your OpenWeatherMap API key to enable live data.</span>
      </div>
    </div>
  )

  const cur = data.current

  return (
    <div className="weather-widget animate-fade-up">
      <div className="weather-header">
        <span><span className="section-icon">🌤</span> Weather — {data.location}</span>
        <span className="rain-badge">
          🌧 {data.rolling_7d_rain_mm?.toFixed(1)} mm / 7 days
        </span>
      </div>

      {/* Current conditions */}
      {cur && (
        <div className="weather-current">
          <div className="weather-temp-block">
            <span className="wx-icon-lg">{wxIcon(cur.description)}</span>
            <div>
              <div className="wx-temp">{toTemp(cur.temp_avg_c)?.toFixed(1)}{unitLabel}</div>
              <div className="wx-desc">{cur.description}</div>
            </div>
          </div>
          <div className="weather-stats">
            <div className="wx-stat">
              <span className="wx-stat-label">High / Low</span>
              <span className="wx-stat-val">{toTemp(cur.temp_max_c)?.toFixed(0)}{unitLabel} / {toTemp(cur.temp_min_c)?.toFixed(0)}{unitLabel}</span>
            </div>
            <div className="wx-stat">
              <span className="wx-stat-label">Humidity</span>
              <span className="wx-stat-val">{cur.humidity_pct?.toFixed(0)}%</span>
            </div>
            <div className="wx-stat">
              <span className="wx-stat-label">Rainfall today</span>
              <span className="wx-stat-val">{cur.rainfall_mm?.toFixed(1)} mm</span>
            </div>
          </div>
        </div>
      )}

      {/* 5-day forecast strip */}
      {data.forecast?.length > 0 && (
        <div className="forecast-strip">
          {data.forecast.slice(0,5).map((day, i) => (
            <div key={i} className="forecast-day">
              <div className="fc-label">
                {i === 0 ? 'Today' : new Date(day.date).toLocaleDateString('en-US',{weekday:'short'})}
              </div>
              <div className="fc-icon">{wxIcon('')}</div>
              <div className="fc-temp">{toTemp(day.tavg)?.toFixed(0)}{unitLabel}</div>
              <div className="fc-rain">{day.precip?.toFixed(0)} mm</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
