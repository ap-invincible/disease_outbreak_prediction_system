import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Landing from './pages/Landing'
import Login from './pages/Login'
import HospitalForum from './pages/HospitalForum'
import Dashboard from './pages/Dashboard'
import Profile from './pages/Profile'

function ProtectedRoute({ children }) {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

function App() {
  return (
    <>
      {/* Background layer - pointer-events-none so it never captures clicks */}
      <div style={{
        position: 'fixed',
        inset: 0,
        zIndex: -1,
        pointerEvents: 'none',
        background: '#0F172A',
        backgroundImage: 'radial-gradient(at 40% 20%, hsla(228,100%,74%,0.15) 0px, transparent 50%), radial-gradient(at 80% 0%, hsla(189,100%,56%,0.15) 0px, transparent 50%), radial-gradient(at 0% 50%, hsla(355,100%,93%,0.1) 0px, transparent 50%)',
      }} />

      {/* App content - normal flow, fully interactive */}
      <div style={{ position: 'relative', zIndex: 1, minHeight: '100vh', color: 'white' }}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/login" element={<Login />} />

            <Route path="/hospital" element={
              <ProtectedRoute>
                <HospitalForum />
              </ProtectedRoute>
            } />

            <Route path="/dashboard" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />

            <Route path="/profile" element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            } />
          </Routes>
        </BrowserRouter>
      </div>
    </>
  )
}

export default App
