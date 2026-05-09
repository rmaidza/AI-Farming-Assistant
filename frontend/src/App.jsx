import { useState, useEffect } from 'react'
import Onboarding from './components/Onboarding.jsx'
import Dashboard from './components/Dashboard.jsx'

export default function App() {
  const [sessionData, setSessionData] = useState(null)
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') !== 'light'
  })
  const [tempUnit, setTempUnit] = useState(() => {
    return localStorage.getItem('tempUnit') || 'C'
  })

  function toggleTempUnit() {
    setTempUnit(u => {
      const next = u === 'C' ? 'F' : 'C'
      localStorage.setItem('tempUnit', next)
      return next
    })
  }

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  function handleSetupComplete(data) {
    setSessionData(data)
  }

  function handleReset() {
    setSessionData(null)
  }

  return sessionData
    ? <Dashboard
        sessionData={sessionData}
        onReset={handleReset}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode(d => !d)}
        tempUnit={tempUnit}
        onToggleTempUnit={toggleTempUnit}
      />
    : <Onboarding
        onSetupComplete={handleSetupComplete}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode(d => !d)}
      />
}
