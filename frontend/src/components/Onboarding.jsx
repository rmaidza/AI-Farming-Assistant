import { useState, useEffect } from 'react'
import './Onboarding.css'

/* Ohio cities with lat/lon pre-filled */
const OHIO_CITIES = [
  { name: 'Columbus',      lat: 39.9612,  lon: -82.9988 },
  { name: 'Cleveland',     lat: 41.4993,  lon: -81.6944 },
  { name: 'Cincinnati',    lat: 39.1031,  lon: -84.5120 },
  { name: 'Toledo',        lat: 41.6639,  lon: -83.5552 },
  { name: 'Akron',         lat: 41.0814,  lon: -81.5190 },
  { name: 'Dayton',        lat: 39.7589,  lon: -84.1916 },
  { name: 'Youngstown',    lat: 41.0998,  lon: -80.6495 },
  { name: 'Canton',        lat: 40.7989,  lon: -81.3784 },
  { name: 'Lorain',        lat: 41.4523,  lon: -82.1824 },
  { name: 'Springfield',   lat: 39.9242,  lon: -83.8088 },
  { name: 'Mansfield',     lat: 40.7584,  lon: -82.5154 },
  { name: 'Wooster',       lat: 40.8051,  lon: -81.9349 },
  { name: 'Findlay',       lat: 41.0442,  lon: -83.6499 },
  { name: 'Lima',          lat: 40.7420,  lon: -84.1052 },
  { name: 'Zanesville',    lat: 39.9403,  lon: -82.0132 },
]

const CROP_META = {
  maize:    { label: 'Maize (Corn)', icon: '🌽', color: '#f5a623', desc: 'Warm-season • ~156 days' },
  soybeans: { label: 'Soybeans',     icon: '🫘', color: '#52b87a', desc: 'Warm-season • ~151 days' },
  wheat:    { label: 'Winter Wheat', icon: '🌾', color: '#c8b99a', desc: 'Cool-season • ~280 days' },
  oats:     { label: 'Oats',         icon: '🌿', color: '#4a9eca', desc: 'Cool-season • ~131 days' },
}

export default function Onboarding({ onSetupComplete, darkMode, onToggleTheme }) {
  const [recommendations, setRecommendations] = useState(null)
  const [loadingRecs, setLoadingRecs]         = useState(true)

  const [city,         setCity]        = useState(OHIO_CITIES[0])
  const [farmSize,     setFarmSize]    = useState('')
  const [selectedCrop, setSelectedCrop] = useState(null)
  const [plantingDate, setPlantingDate] = useState('')
  const [submitting,   setSubmitting]  = useState(false)
  const [error,        setError]       = useState(null)

  // Fetch crop recommendations based on today's date
  useEffect(() => {
    fetch('/api/recommend')
      .then(r => r.json())
      .then(data => {
        setRecommendations(data)
        if (data.recommended_crops?.length) {
          const first = data.recommended_crops[0]
          setSelectedCrop(first)
          // Pre-fill ideal planting date
          const detail = data.crop_details?.find(d => d.crop === first)
          if (detail) setPlantingDate(detail.ideal_planting_date)
        }
      })
      .catch(() => {
        // Fallback: show all crops if API unreachable
        setRecommendations({ recommended_crops: ['maize','soybeans','wheat','oats'], message: '', crop_details: [] })
      })
      .finally(() => setLoadingRecs(false))
  }, [])

  function handleCropSelect(crop) {
    setSelectedCrop(crop)
    const detail = recommendations?.crop_details?.find(d => d.crop === crop)
    if (detail) setPlantingDate(detail.ideal_planting_date)
  }

  function handleCityChange(e) {
    const found = OHIO_CITIES.find(c => c.name === e.target.value)
    if (found) setCity(found)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)

    if (!selectedCrop)    return setError('Please select a crop.')
    if (!farmSize || +farmSize <= 0) return setError('Enter a valid farm size.')
    if (!plantingDate)    return setError('Select a planting date.')

    setSubmitting(true)
    try {
      const resp = await fetch('/api/setup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          farm_name:          `${city.name} Demo Farm`,
          location:           `${city.name}, Ohio`,
          latitude:           city.lat,
          longitude:          city.lon,
          farm_size_hectares: parseFloat(farmSize),
          crop:               selectedCrop,
          planting_date:      plantingDate,
        }),
      })
      if (!resp.ok) {
        const err = await resp.json()
        throw new Error(err.error || 'Setup failed')
      }
      const data = await resp.json()
      onSetupComplete(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const selectedDetail = recommendations?.crop_details?.find(d => d.crop === selectedCrop)

  return (
    <div className="onboarding-root">
      {/* Left panel — branding */}
      <aside className="onboarding-aside">
        <div className="aside-inner">
          <div className="brand">
            <div className="brand-icon">
              <svg viewBox="0 0 40 40" fill="none">
                <circle cx="20" cy="20" r="20" fill="rgba(61,139,95,0.15)"/>
                <path d="M20 8 C14 8 10 13 10 18 C10 24 14 28 20 32 C26 28 30 24 30 18 C30 13 26 8 20 8Z" fill="#3d8b5f" opacity="0.8"/>
                <path d="M20 12 L20 30 M14 18 L20 14 L26 18" stroke="#f0e6c8" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div>
              <div className="brand-name">AI Farming Assistant</div>
              <div className="brand-tagline">Smart crop management for Ohio</div>
            </div>
          </div>

          <div className="aside-hero">
            <h1 className="aside-heading">
              Grow smarter,<br />
              <em>harvest better.</em>
            </h1>
            <p className="aside-subtext">
              Get AI-powered yield predictions, a dynamic growing calendar,
              and real-time weather alerts — all tailored to your farm.
            </p>
          </div>

          <div className="aside-features stagger">
            {[
              { icon: '🌡️', label: 'Live weather alerts' },
              { icon: '📈', label: 'ML yield prediction' },
              { icon: '📅', label: 'Season task calendar' },
              { icon: '🔍', label: 'Pest & disease scouting' },
            ].map(f => (
              <div key={f.label} className="aside-feature animate-fade-up">
                <span className="aside-feature-icon">{f.icon}</span>
                <span>{f.label}</span>
              </div>
            ))}
          </div>

          <div className="aside-crops">
            <div className="aside-crops-label">Supported crops</div>
            <div className="aside-crops-chips">
              {Object.entries(CROP_META).map(([key, meta]) => (
                <span key={key} className="crop-chip" style={{ '--chip-color': meta.color }}>
                  {meta.icon} {meta.label}
                </span>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* Right panel — form */}
      <main className="onboarding-main">
        <form className="onboarding-form animate-fade-up" onSubmit={handleSubmit}>
          <div className="form-header">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <h2 className="form-title">Set up your farm</h2>
                <p className="form-subtitle">Tell us about your operation to get started</p>
              </div>
              <button className="dash-theme-btn" onClick={onToggleTheme} title="Toggle theme">
                {darkMode ? '☀️' : '🌙'}
              </button>
            </div>
          </div>

          {/* Location */}
          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">📍</span> Location
            </label>
            <select className="form-select" value={city.name} onChange={handleCityChange}>
              {OHIO_CITIES.map(c => (
                <option key={c.name} value={c.name}>{c.name}, Ohio</option>
              ))}
            </select>
            <div className="form-hint">
              Lat {city.lat.toFixed(4)}° · Lon {city.lon.toFixed(4)}°
            </div>
          </div>

          {/* Farm size */}
          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">🗺️</span> Farm size
            </label>
            <div className="input-with-unit">
              <input
                type="number"
                className="form-input"
                placeholder="e.g. 5"
                min="0.1"
                step="0.1"
                value={farmSize}
                onChange={e => setFarmSize(e.target.value)}
              />
              <span className="input-unit">hectares</span>
            </div>
          </div>

          {/* Crop selection */}
          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">🌱</span> Choose crop
              {recommendations?.message && (
                <span className="rec-badge">Recommended for today</span>
              )}
            </label>

            {loadingRecs ? (
              <div className="crop-grid">
                {[1,2,3,4].map(i => (
                  <div key={i} className="skeleton" style={{ height: 80 }} />
                ))}
              </div>
            ) : (
              <>
                {recommendations?.message && (
                  <div className="rec-message">{recommendations.message}</div>
                )}
                <div className="crop-grid">
                  {Object.keys(CROP_META).map(crop => {
                    const meta       = CROP_META[crop]
                    const inSeason   = recommendations?.recommended_crops?.includes(crop)
                    const detail     = recommendations?.crop_details?.find(d => d.crop === crop)
                    const isSelected = selectedCrop === crop
                    return (
                      <button
                        key={crop}
                        type="button"
                        className={`crop-btn ${isSelected ? 'crop-btn--selected' : ''} ${!inSeason ? 'crop-btn--off-season' : ''}`}
                        style={{ '--crop-color': meta?.color }}
                        onClick={() => handleCropSelect(crop)}
                      >
                        <span className="crop-btn-icon">{meta?.icon}</span>
                        <span className="crop-btn-label">{meta?.label}</span>
                        <span className="crop-btn-desc">{meta?.desc}</span>
                        {inSeason ? (
                          <span className="crop-btn-season crop-btn-season--in">✅ In season</span>
                        ) : (
                          <span className="crop-btn-season crop-btn-season--out">🕐 Out of season</span>
                        )}
                        {detail && (
                          <span className="crop-btn-harvest">
                            Harvest ~{new Date(detail.estimated_harvest_date).toLocaleDateString('en-US',{month:'short',day:'numeric'})}
                          </span>
                        )}
                        {isSelected && <span className="crop-btn-check">✓</span>}
                      </button>
                    )
                  })}
                </div>
              </>
            )}
          </div>

          {/* Planting date */}
          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">📅</span> Planting date
            </label>
            <input
              type="date"
              className="form-input"
              value={plantingDate}
              onChange={e => setPlantingDate(e.target.value)}
            />
            {selectedDetail && (
              <div className="form-hint">
                Ideal window: {new Date(selectedDetail.planting_window_start).toLocaleDateString('en-US',{month:'short',day:'numeric'})}
                {' → '}
                {new Date(selectedDetail.planting_window_end).toLocaleDateString('en-US',{month:'short',day:'numeric'})}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="form-error">
              <span>⚠️</span> {error}
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            className={`submit-btn ${submitting ? 'submit-btn--loading' : ''}`}
            disabled={submitting}
          >
            {submitting ? (
              <>
                <span className="spinner" />
                Setting up your farm…
              </>
            ) : (
              <>
                Launch Dashboard
                <span className="submit-arrow">→</span>
              </>
            )}
          </button>
        </form>
      </main>
    </div>
  )
}
