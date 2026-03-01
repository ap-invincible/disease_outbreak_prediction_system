import React, { useState, useEffect } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

export default function Login() {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [pin, setPin] = useState('');
    const [password, setPassword] = useState('');
    const [regStep, setRegStep] = useState('email'); // 'email' | 'pin' | 'details'
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login, register, user } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const nextPath = new URLSearchParams(location.search).get('next') || '/dashboard';

    useEffect(() => {
        if (user) {
            navigate(nextPath, { replace: true });
        }
    }, [user, navigate, nextPath]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        if (isLogin) {
            const result = await login(username, password);
            if (result.success) navigate(nextPath);
            else setError(result.error);
            setLoading(false);
            return;
        }

        try {
            if (regStep === 'email') {
                await axios.post('http://localhost:5000/api/auth/request-pin', { email });
                setRegStep('pin');
            } else if (regStep === 'pin') {
                await axios.post('http://localhost:5000/api/auth/verify-pin', { email, pin });
                setRegStep('details');
            } else if (regStep === 'details') {
                const result = await register(username, email, password);
                if (result.success) navigate(nextPath);
                else setError(result.error);
            }
        } catch (err) {
            setError(err.response?.data?.error || 'An error occurred during verification');
        }

        setLoading(false);
    };

    return (
        <div className="relative" style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, paddingBottom: 60, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
            <div className="bg-blobs"></div>
            <div className="glass-panel" style={{ padding: '3rem', width: '100%', maxWidth: 400, position: 'relative', zIndex: 10 }}>

                {/* Minimal Icon */}
                <div style={{ width: 56, height: 56, background: 'rgba(16, 185, 129, 0.1)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 24px', fontSize: 24, border: '1px solid rgba(16, 185, 129, 0.2)', boxShadow: 'inset 0 2px 10px rgba(16, 185, 129, 0.05)' }}>
                    🏥
                </div>

                <h2 style={{ textAlign: 'center', fontSize: 24, fontWeight: 700, marginBottom: 8, letterSpacing: '-0.02em', color: '#0f172a' }}>
                    {isLogin ? 'Welcome Back' : (regStep === 'email' ? 'Create Account' : regStep === 'pin' ? 'Verify Email' : 'Account Details')}
                </h2>
                <p style={{ textAlign: 'center', color: '#64748b', marginBottom: 32, fontSize: 13, fontWeight: 500 }}>
                    {isLogin ? 'Sign in to access the health portal' : (regStep === 'email' ? 'Enter your email to receive a secure pin' : regStep === 'pin' ? 'Enter the 6-digit pin sent to your email' : 'Choose your username and password')}
                </p>

                {error && (
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 12, padding: '10px 14px', marginBottom: 20, fontSize: 13, color: '#b91c1c', textAlign: 'center', backdropFilter: 'blur(4px)' }}>
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit}>
                    {(isLogin || (!isLogin && regStep === 'details')) && (
                        <div style={{ marginBottom: 16 }}>
                            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8, paddingLeft: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Username</label>
                            <input
                                type="text" value={username} required placeholder="username"
                                className="glass-input"
                                onChange={e => setUsername(e.target.value)}
                            />
                        </div>
                    )}

                    {!isLogin && regStep === 'email' && (
                        <div style={{ marginBottom: 28 }}>
                            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8, paddingLeft: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Email Address</label>
                            <input
                                type="email" value={email} required placeholder="doctor@hospital.com"
                                className="glass-input"
                                onChange={e => setEmail(e.target.value)}
                            />
                        </div>
                    )}

                    {!isLogin && regStep === 'pin' && (
                        <div style={{ marginBottom: 28 }}>
                            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8, paddingLeft: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Verification Code</label>
                            <input
                                type="text" value={pin} required placeholder="123456"
                                maxLength={6}
                                className="glass-input"
                                style={{ letterSpacing: '0.5em', textAlign: 'center', fontSize: '1.2rem', fontWeight: 600 }}
                                onChange={e => setPin(e.target.value)}
                            />
                        </div>
                    )}

                    {(isLogin || (!isLogin && regStep === 'details')) && (
                        <div style={{ marginBottom: 28 }}>
                            <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#475569', marginBottom: 8, paddingLeft: 4, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Password</label>
                            <input
                                type="password" value={password} required placeholder="••••••••"
                                className="glass-input"
                                onChange={e => setPassword(e.target.value)}
                            />
                        </div>
                    )}

                    {/* Primary Button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="glass-button"
                        style={{ width: '100%', padding: '14px', fontSize: 14, fontWeight: 600, cursor: loading ? 'default' : 'pointer', letterSpacing: '0.02em' }}
                    >
                        {loading ? 'Processing...' : isLogin ? 'Sign In' : (regStep === 'email' ? 'Send PIN' : regStep === 'pin' ? 'Verify PIN' : 'Complete Registration')}
                    </button>

                    {!isLogin && regStep !== 'email' && (
                        <button
                            type="button"
                            onClick={() => setRegStep('email')}
                            style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontWeight: 500, padding: 0, marginTop: 16, display: 'block', width: '100%', fontSize: 13 }}
                        >
                            ← Back to Email
                        </button>
                    )}
                </form>

                <div style={{ marginTop: 24, textAlign: 'center', fontSize: 13, color: '#64748b' }}>
                    {isLogin ? "Don't have an account? " : 'Already registered? '}
                    <button
                        onClick={() => { setIsLogin(!isLogin); setRegStep('email'); setError(''); }}
                        style={{ background: 'none', border: 'none', color: '#10b981', cursor: 'pointer', fontWeight: 600, padding: 0, textDecoration: 'underline', textUnderlineOffset: 4 }}
                    >
                        {isLogin ? 'Register' : 'Sign In'}
                    </button>
                </div>

                <div style={{ marginTop: 32, textAlign: 'center' }}>
                    <Link to="/" style={{ color: '#94a3b8', fontSize: 12, textDecoration: 'none', transition: 'color 0.2s', fontWeight: 500 }} onMouseOver={e => e.target.style.color = '#475569'} onMouseOut={e => e.target.style.color = '#94a3b8'}>
                        ← Back to Home
                    </Link>
                </div>
            </div>
        </div>
    );
}
