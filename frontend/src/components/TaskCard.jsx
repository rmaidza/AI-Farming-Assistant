import './TaskCard.css'

const TASK_ICONS = {
  planting:   '🌱',
  fertilizer: '🟡',
  irrigation: '💧',
  scouting:   '🔍',
  harvest:    '🌾',
  alert:      '⚠️',
  milestone:  '🏁',
  herbicide:  '🌿',
  fungicide:  '🍄',
}

const STAGE_LABELS = {
  germination:  'Germination',
  vegetative:   'Vegetative',
  reproductive: 'Reproductive',
}

const PRIORITY_META = {
  normal:   { label: 'Normal',   cls: '' },
  warning:  { label: 'Warning',  cls: 'task-card--warning' },
  critical: { label: 'Critical', cls: 'task-card--critical' },
}

function formatDate(dateStr) {
  const d = new Date(dateStr)
  // Offset timezone issue with date-only strings
  const local = new Date(d.getTime() + d.getTimezoneOffset() * 60000)
  return local.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })
}

function isToday(dateStr) {
  const today = new Date()
  const d     = new Date(dateStr)
  const local = new Date(d.getTime() + d.getTimezoneOffset() * 60000)
  return local.toDateString() === today.toDateString()
}

function isPast(dateStr) {
  const d     = new Date(dateStr)
  const local = new Date(d.getTime() + d.getTimezoneOffset() * 60000)
  return local < new Date(new Date().toDateString())
}

export default function TaskCard({ task, onComplete, completing }) {
  const icon     = TASK_ICONS[task.task_type]  || '📋'
  const stage    = STAGE_LABELS[task.stage_code] || task.stage_code
  const priority = PRIORITY_META[task.priority] || PRIORITY_META.normal
  const today    = isToday(task.task_date)
  const past     = isPast(task.task_date)

  return (
    <div className={`task-card ${task.is_alert ? 'task-card--alert' : ''} ${priority.cls} ${task.is_completed ? 'task-card--done' : ''} animate-fade-up`}>
      {/* Left accent bar */}
      <div className="task-accent" />

      <div className="task-body">
        {/* Top row */}
        <div className="task-top">
          <div className="task-meta-row">
            <span className="task-icon">{icon}</span>
            <span className="task-stage">{stage}</span>
            {task.is_alert && <span className="task-alert-badge">ALERT</span>}
            {today && !task.is_completed && <span className="task-today-badge">Today</span>}
            {past && !task.is_completed && !today && <span className="task-overdue-badge">Overdue</span>}
          </div>
          <div className="task-date">{formatDate(task.task_date)}</div>
        </div>

        {/* Title */}
        <div className="task-title">{task.task_title}</div>

        {/* Description */}
        {task.task_description && (
          <p className="task-desc">{task.task_description}</p>
        )}

        {/* Footer */}
        <div className="task-footer">
          <span className={`task-type-pill task-type-pill--${task.task_type}`}>
            {task.task_type}
          </span>

          {task.is_completed ? (
            <span className="task-completed-label">
              ✓ Completed {task.completed_date ? formatDate(task.completed_date) : ''}
            </span>
          ) : (
            <button
              className={`task-complete-btn ${completing ? 'task-complete-btn--loading' : ''}`}
              onClick={() => onComplete(task.id)}
              disabled={completing}
            >
              {completing
                ? <span className="spinner-sm" />
                : <><span className="check-box" />Mark complete</>
              }
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
