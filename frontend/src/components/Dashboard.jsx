import { useState, useEffect, useCallback } from 'react'
import WeatherWidget from './WeatherWidget.jsx'
import TaskCard from './TaskCard.jsx'
import './Dashboard.css'

const CROP_META = {
  maize:    { label: 'Maize', icon: '🌽', color: '#f5a623' },
  soybeans: { label: 'Soybeans', icon: '🫘', color: '#52b87a' },
  wheat:    { label: 'Winter Wheat', icon: '🌾', color: '#c8b99a' },
  oats:     { label: 'Oats', icon: '🌿', color: '#4a9eca' },
}

const STAGE_PROGRESS = {
  germination:  33,
  vegetative:   66,
  reproductive: 100,
}

const TASK_TYPE_COLORS = {
  planting:   '#52b87a',
  fertilizer: '#f5a623',
  irrigation: '#4a9eca',
  scouting:   '#a78bfa',
  harvest:    '#c8b99a',
  alert:      '#e05454',
  milestone:  '#3d8b5f',
  herbicide:  '#fb923c',
  fungicide:  '#f472b6',
}

export default function Dashboard({ sessionData, onReset, darkMode, onToggleTheme, tempUnit = 'C', onToggleTempUnit }) {
  const [dashboard, setDashboard]       = useState(null)
  const [allTasks,  setAllTasks]        = useState([])
  const [loading,   setLoading]         = useState(true)
  const [completing, setCompleting]     = useState(null)
  const [showDone,  setShowDone]        = useState(false)
  const [alertDismissed, setAlertDismissed] = useState(false)
  const [activeTab, setActiveTab]       = useState('dashboard')
  const [calMonth,  setCalMonth]        = useState(new Date())

  const fetchDashboard = useCallback(() => {
    fetch('/api/dashboard')
      .then(r => r.json())
      .then(data => {
        setDashboard(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  // Fetch ALL tasks for calendar and analytics views
  const fetchAllTasks = useCallback(() => {
    fetch('/api/tasks/all')
      .then(r => r.json())
      .then(data => setAllTasks(data.tasks || []))
      .catch(() => {})
  }, [])

  useEffect(() => {
    fetchDashboard()
    fetchAllTasks()
    const interval = setInterval(fetchDashboard, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [fetchDashboard, fetchAllTasks])

  async function handleComplete(taskId) {
    setCompleting(taskId)
    try {
      await fetch(`/api/tasks/${taskId}/complete`, { method: 'POST' })
      fetchDashboard()
      fetchAllTasks()
    } finally {
      setCompleting(null)
    }
  }

  const crop    = dashboard?.crop
  const meta    = CROP_META[crop] || {}
  const yield_  = sessionData?.predicted_yield?.yield_per_ha
  const yieldNote = sessionData?.predicted_yield?.note
  const alerts  = dashboard?.alerts || []
  const tasks   = dashboard?.tasks  || []
  const done    = dashboard?.completed_tasks || []
  const stageProgress = STAGE_PROGRESS[dashboard?.current_stage] || 33
  const harvestDate = dashboard?.expected_harvest_date
    ? new Date(dashboard.expected_harvest_date).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
    : null

  // ── Calendar helpers ──────────────────────────────────────
  function getDaysInMonth(date) {
    return new Date(date.getFullYear(), date.getMonth() + 1, 0).getDate()
  }
  function getFirstDayOfMonth(date) {
    return new Date(date.getFullYear(), date.getMonth(), 1).getDay()
  }
  function getTasksForDay(day) {
    const d = new Date(calMonth.getFullYear(), calMonth.getMonth(), day)
    const iso = d.toISOString().split('T')[0]
    return allTasks.filter(t => t.task_date === iso)
  }

  // ── Analytics helpers ─────────────────────────────────────
  const completedCount = done.length
  const pendingCount   = tasks.length + alerts.length
  const totalCount     = completedCount + pendingCount
  const completionPct  = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0

  const tasksByType = {}
  allTasks.forEach(t => {
    const type = t.task_type || 'other'
    tasksByType[type] = (tasksByType[type] || 0) + 1
  })

  const tasksByStage = {}
  allTasks.forEach(t => {
    const stage = t.stage_code || 'unknown'
    tasksByStage[stage] = (tasksByStage[stage] || 0) + 1
  })

  return (
    <div className="dashboard-root">
      {/* ── Top Navigation Bar ───────────────────────────────── */}
      <header className="dash-header">
        <div className="dash-brand">
          <div className="dash-brand-icon">
            <svg viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="16" fill="rgba(61,139,95,0.2)"/>
              <path d="M16 6 C11 6 8 10 8 14 C8 19 11 22 16 26 C21 22 24 19 24 14 C24 10 21 6 16 6Z" fill="#3d8b5f"/>
              <path d="M16 10 L16 24 M11 15 L16 11 L21 15" stroke="#f0e6c8" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <div>
            <div className="dash-brand-name">AI Farming Assistant</div>
            <div className="dash-brand-sub">Smart crop management powered by AI</div>
          </div>
        </div>

        <nav className="dash-nav">
          <button
            className={`dash-nav-btn ${activeTab === 'dashboard' ? 'dash-nav-btn--active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >Dashboard</button>
          <button
            className={`dash-nav-btn ${activeTab === 'calendar' ? 'dash-nav-btn--active' : ''}`}
            onClick={() => setActiveTab('calendar')}
          >Calendar</button>
          <button
            className={`dash-nav-btn ${activeTab === 'analytics' ? 'dash-nav-btn--active' : ''}`}
            onClick={() => setActiveTab('analytics')}
          >Analytics</button>
        </nav>

        <div className="dash-header-right">
          <button className="dash-theme-btn" onClick={onToggleTempUnit} title="Toggle temperature unit">
            °{tempUnit === 'C' ? 'F' : 'C'}
          </button>
          <button className="dash-theme-btn" onClick={onToggleTheme} title="Toggle theme">
            {darkMode ? '☀️' : '🌙'}
          </button>
          {!loading && crop && (
            <div className="dash-crop-badge" style={{ '--crop-color': meta.color }}>
              {meta.icon} {meta.label}
            </div>
          )}
          <button className="dash-reset-btn" onClick={onReset} title="Start over">
            ↺ Reset
          </button>
        </div>
      </header>

      {loading ? (
        <div className="dash-loading">
          <div className="dash-loading-spinner" />
          <p>Loading your farm dashboard…</p>
        </div>
      ) : (
        <div className="dash-body">

          {/* ══════════════════════════════════════════════════
              DASHBOARD TAB
          ══════════════════════════════════════════════════ */}
          {activeTab === 'dashboard' && (
            <>
              {alerts.length > 0 && !alertDismissed && (
                <div className={`alert-banner alert-banner--${alerts[0]?.priority === 'critical' ? 'critical' : 'warning'} animate-fade-up`}>
                  <div className="alert-banner-left">
                    <span className="alert-banner-icon">
                      {alerts[0]?.priority === 'critical' ? '🚨' : '⚠️'}
                    </span>
                    <div>
                      <div className="alert-banner-title">
                        {alerts.length === 1 ? alerts[0].task_title : `${alerts.length} active alerts require attention`}
                      </div>
                      {alerts.length === 1 && (
                        <div className="alert-banner-desc">{alerts[0].task_description}</div>
                      )}
                    </div>
                  </div>
                  <button className="alert-dismiss" onClick={() => setAlertDismissed(true)}>✕</button>
                </div>
              )}

              <div className="stat-grid stagger">
                <div className="stat-card animate-fade-up">
                  <div className="stat-label">Current Stage</div>
                  <div className="stat-value stat-value--stage">
                    {dashboard?.current_stage
                      ? dashboard.current_stage.charAt(0).toUpperCase() + dashboard.current_stage.slice(1)
                      : '—'}
                  </div>
                  <div className="stage-progress-bar">
                    <div className="stage-progress-fill" style={{ width: `${stageProgress}%` }} />
                  </div>
                  <div className="stage-progress-labels">
                    <span>Germination</span>
                    <span>Vegetative</span>
                    <span>Reproductive</span>
                  </div>
                </div>

                <div className="stat-card animate-fade-up">
                  <div className="stat-label">Days from Planting</div>
                  <div className="stat-value">
                    {dashboard?.days_from_planting ?? '—'}
                    <span className="stat-unit"> days</span>
                  </div>
                  {dashboard?.stage_info && (
                    <div className="stat-sub">{dashboard.stage_info.stage_name}</div>
                  )}
                </div>

                <div className="stat-card animate-fade-up">
                  <div className="stat-label">Predicted Yield</div>
                  {yield_ ? (
                    <>
                      <div className="stat-value">
                        {yield_.toFixed(1)}
                        <span className="stat-unit"> bu/ha</span>
                      </div>
                      <div className="stat-sub">
                        {sessionData?.predicted_yield?.total_yield_bu?.toFixed(0)} bu total
                      </div>
                    </>
                  ) : (
                    <div className="stat-value stat-value--unavail">—</div>
                  )}
                  {!yield_ && (
                    <div className="stat-sub stat-sub--muted">Model loading…</div>
                  )}
                </div>

                <div className="stat-card animate-fade-up">
                  <div className="stat-label">Expected Harvest</div>
                  <div className="stat-value stat-value--sm">{harvestDate || '—'}</div>
                  {dashboard?.stage_info && (
                    <div className="stat-sub">
                      Opt temp: {tempUnit === 'C'
                        ? `${dashboard.stage_info.temp_min_c}–${dashboard.stage_info.temp_max_c}°C`
                        : `${(dashboard.stage_info.temp_min_c * 9/5 + 32).toFixed(0)}–${(dashboard.stage_info.temp_max_c * 9/5 + 32).toFixed(0)}°F`}
                    </div>
                  )}
                </div>
              </div>

              <div className="dash-grid">
                <section className="tasks-section">
                  <div className="section-header">
                    <h2 className="section-title">
                      <span className="section-icon">📅</span>
                      Upcoming Tasks
                      <span className="task-count">{tasks.length}</span>
                    </h2>
                  </div>

                  {tasks.length === 0 ? (
                    <div className="tasks-empty">
                      <span>🌱</span>
                      <p>No upcoming tasks in the next 14 days.</p>
                    </div>
                  ) : (
                    <div className="task-list stagger">
                      {tasks.map(task => (
                        <TaskCard key={task.id} task={task} onComplete={handleComplete} completing={completing === task.id} />
                      ))}
                    </div>
                  )}

                  {alerts.length > 0 && (
                    <div className="alerts-section">
                      <div className="section-header">
                        <h2 className="section-title section-title--alert">
                          <span className="section-icon">⚠️</span>
                          Active Alerts
                          <span className="task-count task-count--alert">{alerts.length}</span>
                        </h2>
                      </div>
                      <div className="task-list">
                        {alerts.map(task => (
                          <TaskCard key={task.id} task={task} onComplete={handleComplete} completing={completing === task.id} />
                        ))}
                      </div>
                    </div>
                  )}

                  {done.length > 0 && (
                    <div className="done-section">
                      <button className="done-toggle" onClick={() => setShowDone(v => !v)}>
                        <span>{showDone ? '▾' : '▸'}</span>
                        Completed Tasks
                        <span className="task-count">{done.length}</span>
                      </button>
                      {showDone && (
                        <div className="task-list">
                          {done.map(task => (
                            <TaskCard key={task.id} task={task} onComplete={handleComplete} completing={completing === task.id} />
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </section>

                <aside className="dash-aside">
                  <WeatherWidget key={sessionData?.farm_id || 'weather'} tempUnit={tempUnit} />
                  {dashboard?.stage_info?.description && (
                    <div className="stage-info-card animate-fade-up">
                      <div className="stage-info-header">
                        <span className="section-icon">🌿</span>
                        <span>Stage Overview</span>
                        <span className="stage-info-badge">{dashboard.current_stage}</span>
                      </div>
                      <p className="stage-info-desc">{dashboard.stage_info.description}</p>
                      <div className="stage-info-thresholds">
                        <div className="threshold">
                          <span className="threshold-icon">🌡️</span>
                          <span className="threshold-label">Optimal temp</span>
                          <span className="threshold-val">
                            {tempUnit === 'C'
                              ? `${dashboard.stage_info.temp_min_c}–${dashboard.stage_info.temp_max_c}°C`
                              : `${(dashboard.stage_info.temp_min_c * 9/5 + 32).toFixed(0)}–${(dashboard.stage_info.temp_max_c * 9/5 + 32).toFixed(0)}°F`}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  {yieldNote && (
                    <div className="yield-note animate-fade-up">
                      <div className="yield-note-header"><span>📈</span> Yield Forecast</div>
                      <p>{yieldNote}</p>
                    </div>
                  )}
                </aside>
              </div>
            </>
          )}

          {/* ══════════════════════════════════════════════════
              CALENDAR TAB
          ══════════════════════════════════════════════════ */}
          {activeTab === 'calendar' && (
            <div className="cal-container animate-fade-up">
              <div className="cal-header">
                <button className="cal-nav-btn" onClick={() => setCalMonth(m => new Date(m.getFullYear(), m.getMonth() - 1))}>‹</button>
                <h2 className="cal-month-title">
                  {calMonth.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                </h2>
                <button className="cal-nav-btn" onClick={() => setCalMonth(m => new Date(m.getFullYear(), m.getMonth() + 1))}>›</button>
                <button className="cal-today-btn" onClick={() => setCalMonth(new Date())}>Today</button>
              </div>

              <div className="cal-legend">
                {Object.entries(TASK_TYPE_COLORS).slice(0, 6).map(([type, color]) => (
                  <span key={type} className="cal-legend-item">
                    <span className="cal-legend-dot" style={{ background: color }} />
                    {type}
                  </span>
                ))}
              </div>

              <div className="cal-grid">
                {['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].map(d => (
                  <div key={d} className="cal-day-header">{d}</div>
                ))}
                {Array.from({ length: getFirstDayOfMonth(calMonth) }).map((_, i) => (
                  <div key={`empty-${i}`} className="cal-cell cal-cell--empty" />
                ))}
                {Array.from({ length: getDaysInMonth(calMonth) }).map((_, i) => {
                  const day = i + 1
                  const dayTasks = getTasksForDay(day)
                  const isToday =
                    new Date().getDate() === day &&
                    new Date().getMonth() === calMonth.getMonth() &&
                    new Date().getFullYear() === calMonth.getFullYear()
                  return (
                    <div key={day} className={`cal-cell ${isToday ? 'cal-cell--today' : ''} ${dayTasks.length > 0 ? 'cal-cell--has-tasks' : ''}`}>
                      <span className="cal-day-num">{day}</span>
                      <div className="cal-task-dots">
                        {dayTasks.slice(0, 3).map(t => (
                          <span
                            key={t.id}
                            className="cal-task-dot"
                            style={{ background: TASK_TYPE_COLORS[t.task_type] || '#3d8b5f' }}
                            title={t.task_title}
                          />
                        ))}
                        {dayTasks.length > 3 && <span className="cal-task-more">+{dayTasks.length - 3}</span>}
                      </div>
                      {dayTasks.length > 0 && (
                        <div className="cal-task-labels">
                          {dayTasks.slice(0, 2).map(t => (
                            <div
                              key={t.id}
                              className={`cal-task-label ${t.is_completed ? 'cal-task-label--done' : ''}`}
                              style={{ '--task-color': TASK_TYPE_COLORS[t.task_type] || '#3d8b5f' }}
                            >
                              {t.task_title}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════
              ANALYTICS TAB
          ══════════════════════════════════════════════════ */}
          {activeTab === 'analytics' && (
            <div className="analytics-container animate-fade-up">
              <h2 className="analytics-title">Season Analytics</h2>
              <p className="analytics-sub">Overview of your {meta.label || 'crop'} season progress</p>

              {/* Stat summary cards */}
              <div className="analytics-stat-row">
                <div className="analytics-stat-card">
                  <div className="analytics-stat-icon">✅</div>
                  <div className="analytics-stat-val">{completedCount}</div>
                  <div className="analytics-stat-label">Tasks Completed</div>
                </div>
                <div className="analytics-stat-card">
                  <div className="analytics-stat-icon">📋</div>
                  <div className="analytics-stat-val">{pendingCount}</div>
                  <div className="analytics-stat-label">Tasks Pending</div>
                </div>
                <div className="analytics-stat-card">
                  <div className="analytics-stat-icon">📊</div>
                  <div className="analytics-stat-val">{completionPct}%</div>
                  <div className="analytics-stat-label">Completion Rate</div>
                </div>
                <div className="analytics-stat-card">
                  <div className="analytics-stat-icon">🌾</div>
                  <div className="analytics-stat-val">{yield_ ? yield_.toFixed(1) : '—'}</div>
                  <div className="analytics-stat-label">Predicted bu/ha</div>
                </div>
              </div>

              {/* Completion progress bar */}
              <div className="analytics-section">
                <h3 className="analytics-section-title">Task Completion Progress</h3>
                <div className="analytics-progress-bar-wrap">
                  <div className="analytics-progress-bar">
                    <div
                      className="analytics-progress-fill"
                      style={{ width: `${completionPct}%` }}
                    />
                  </div>
                  <span className="analytics-progress-label">{completionPct}% complete</span>
                </div>
              </div>

              {/* Tasks by type bar chart */}
              <div className="analytics-section">
                <h3 className="analytics-section-title">Tasks by Type</h3>
                <div className="analytics-bar-chart">
                  {Object.entries(tasksByType).map(([type, count]) => {
                    const pct = Math.round((count / allTasks.length) * 100)
                    return (
                      <div key={type} className="analytics-bar-row">
                        <div className="analytics-bar-label">{type}</div>
                        <div className="analytics-bar-track">
                          <div
                            className="analytics-bar-fill"
                            style={{
                              width: `${pct}%`,
                              background: TASK_TYPE_COLORS[type] || '#3d8b5f'
                            }}
                          />
                        </div>
                        <div className="analytics-bar-count">{count}</div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Tasks by stage */}
              <div className="analytics-section">
                <h3 className="analytics-section-title">Tasks by Growth Stage</h3>
                <div className="analytics-stage-cards">
                  {Object.entries(tasksByStage).map(([stage, count]) => (
                    <div key={stage} className="analytics-stage-card">
                      <div className="analytics-stage-name">{stage}</div>
                      <div className="analytics-stage-count">{count} tasks</div>
                      <div className="analytics-stage-bar">
                        <div
                          className="analytics-stage-fill"
                          style={{ width: `${Math.round((count / allTasks.length) * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Season info */}
              <div className="analytics-section">
                <h3 className="analytics-section-title">Season Summary</h3>
                <div className="analytics-info-grid">
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Crop</span>
                    <span className="analytics-info-val">{meta.icon} {meta.label}</span>
                  </div>
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Current Stage</span>
                    <span className="analytics-info-val">{dashboard?.current_stage || '—'}</span>
                  </div>
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Days from Planting</span>
                    <span className="analytics-info-val">{dashboard?.days_from_planting ?? '—'} days</span>
                  </div>
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Expected Harvest</span>
                    <span className="analytics-info-val">{harvestDate || '—'}</span>
                  </div>
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Predicted Yield</span>
                    <span className="analytics-info-val">{yield_ ? `${yield_.toFixed(1)} bu/ha` : 'N/A'}</span>
                  </div>
                  <div className="analytics-info-item">
                    <span className="analytics-info-label">Active Alerts</span>
                    <span className="analytics-info-val" style={{ color: alerts.length > 0 ? '#e05454' : '#52b87a' }}>
                      {alerts.length > 0 ? `${alerts.length} alert${alerts.length > 1 ? 's' : ''}` : 'None'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}
