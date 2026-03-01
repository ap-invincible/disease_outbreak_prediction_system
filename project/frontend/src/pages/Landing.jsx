import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ShieldAlert, MapPin, TrendingUp, Clock, Zap, Lock, User as UserIcon } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const features = [
    {
        icon: <Activity className="w-6 h-6 text-blue-400" />,
        bg: 'rgba(59,130,246,0.15)',
        title: 'AI Symptom Detection',
        desc: 'Identify Covid-19, Dengue, Influenza, Typhoid and Cholera from patient symptom profiles using our multi-class Random Forest model.',
    },
    {
        icon: <TrendingUp className="w-6 h-6 text-green-400" />,
        bg: 'rgba(34,197,94,0.15)',
        title: '48-Hour Outbreak Forecast',
        desc: 'Disease-specific trend models predict growth rate and doubling time to warn of outbreaks up to 48 hours in advance.',
    },
    {
        icon: <MapPin className="w-6 h-6 text-red-400" />,
        bg: 'rgba(239,68,68,0.15)',
        title: 'Geographical Heatmaps',
        desc: 'Live Google Maps integration plots outbreak epicenters color-coded by severity — Low, Medium, and High risk zones.',
    },
    {
        icon: <ShieldAlert className="w-6 h-6 text-purple-400" />,
        bg: 'rgba(139,92,246,0.15)',
        title: 'Vulnerability Scoring',
        desc: 'Each reported location is scored 0–100 for geographic vulnerability, enabling authorities to prioritise high-risk regions.',
    },
    {
        icon: <Clock className="w-6 h-6 text-yellow-400" />,
        bg: 'rgba(234,179,8,0.15)',
        title: 'Real-Time Dashboard',
        desc: 'Live analytics feed showing all active detections, risk levels, and key metrics across every tracked jurisdiction.',
    },
    {
        icon: <Lock className="w-6 h-6 text-gray-400" />,
        bg: 'rgba(156,163,175,0.15)',
        title: 'Secure Access Control',
        desc: 'JWT-based authentication backed by a SQLite database ensures only certified healthcare professionals can submit data.',
    },
];

export default function Landing() {
    const { user } = useAuth();

    return (
        <div className="relative" style={{ fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', minHeight: '100vh', paddingBottom: 60 }}>
            <div className="bg-blobs"></div>

            {/* Top Right Login / Profile */}
            <div style={{ position: 'absolute', top: 24, right: 24, zIndex: 50 }}>
                {user ? (
                    <Link to="/profile" style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(255,255,255,0.7)', border: '1px solid #1ecf94ff', color: '#16ad7bff', padding: '10px 20px', borderRadius: 99, fontWeight: 600, fontSize: 14, textDecoration: 'none', backdropFilter: 'blur(12px)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                        <UserIcon className="w-4 h-4" />
                        {user.username}
                    </Link>
                ) : (
                    <Link to="/login" style={{ display: 'inline-block', background: 'rgba(255,255,255,0.5)', border: '1px solid #1ecf94ff', color: '#16ad7bff', padding: '10px 24px', borderRadius: 16, fontWeight: 600, fontSize: 14, textDecoration: 'none', backdropFilter: 'blur(8px)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                        Login
                    </Link>
                )}
            </div>

            {/* ── Hero ── */}
            <section style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: '80px 24px 60px' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: 99, padding: '6px 16px', marginBottom: 24, fontSize: 13, color: '#059669', fontWeight: 600, backdropFilter: 'blur(4px)' }}>
                    <Zap style={{ width: 14, height: 14 }} /> Powered by Multi-Model ML Pipeline
                </div>
                <h1 style={{ fontSize: 'clamp(2.5rem, 6vw, 4.5rem)', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.1, marginBottom: 8, background: 'linear-gradient(to right, #10b981, #0ea5e9)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                    invncble
                </h1>
                <h2 style={{ fontSize: 'clamp(1.25rem, 3vw, 1.75rem)', fontWeight: 600, color: '#10b981', marginBottom: 24, letterSpacing: '-0.01em' }}>
                    Disease outbreak prediction system
                </h2>
                <p style={{ fontSize: 18, color: '#475569', maxWidth: 600, lineHeight: 1.7, marginBottom: 40, fontWeight: 500 }}>
                    An integrated intelligence platform for hospitals and public health agencies to detect, monitor, and respond to disease outbreaks — before they escalate.
                </p>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
                    <Link to={user ? "/hospital" : "/login?next=/hospital"} className="glass-button" style={{ padding: '14px 32px', fontWeight: 600, fontSize: 16, textDecoration: 'none' }}>
                        Get Started →
                    </Link>
                    <Link to={user ? "/dashboard" : "/login?next=/dashboard"} style={{ background: 'rgba(255,255,255,0.5)', border: '1px solid #1ecf94ff', color: '#16ad7bff', padding: '14px 32px', borderRadius: 16, fontWeight: 600, fontSize: 16, textDecoration: 'none', backdropFilter: 'blur(8px)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                        View Dashboard
                    </Link>
                </div>
            </section>

            {/* ── Routes ── */}
            <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24, maxWidth: 900, margin: '0 auto', padding: '0 24px 64px', position: 'relative', zIndex: 10 }}>
                <Link to={user ? "/hospital" : "/login?next=/hospital"} style={{ textDecoration: 'none', color: 'inherit' }}>
                    <div className="glass-panel" style={{ padding: '2rem', cursor: 'pointer', height: '100%' }}>
                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', width: 52, height: 52, borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20, border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                            <Activity style={{ width: 26, height: 26, color: '#10b981' }} />
                        </div>
                        <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 10, color: '#0f172a' }}>Hospital Forum</h2>
                        <p style={{ color: '#475569', lineHeight: 1.6, marginBottom: 20, fontWeight: 500 }}>
                            Submit patient symptom logs, location coordinates, and clinical data. The integrated ML pipeline instantly returns a disease prediction, risk level, and growth rate estimate.
                        </p>
                        <span style={{ color: '#10b981', fontWeight: 600 }}>Enter Forum →</span>
                    </div>
                </Link>

                <Link to={user ? "/dashboard" : "/login?next=/dashboard"} style={{ textDecoration: 'none', color: 'inherit' }}>
                    <div className="glass-panel" style={{ padding: '2rem', cursor: 'pointer', height: '100%' }}>
                        <div style={{ background: 'rgba(14, 165, 233, 0.1)', width: 52, height: 52, borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20, border: '1px solid rgba(14, 165, 233, 0.2)' }}>
                            <ShieldAlert style={{ width: 26, height: 26, color: '#0ea5e9' }} />
                        </div>
                        <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 10, color: '#0f172a' }}>Primary Dashboard</h2>
                        <p style={{ color: '#475569', lineHeight: 1.6, marginBottom: 20, fontWeight: 500 }}>
                            Command-center view of all active outbreaks. Track growth rates, doubling times, vulnerability scores, and geographic hotspot heatmaps in real time.
                        </p>
                        <span style={{ color: '#0ea5e9', fontWeight: 600 }}>View Dashboard →</span>
                    </div>
                </Link>
            </section>

            {/* ── Features ── */}
            <section style={{ maxWidth: 1100, margin: '0 auto', padding: '0 24px 80px', position: 'relative', zIndex: 10 }}>
                <h2 style={{ textAlign: 'center', fontSize: 32, fontWeight: 700, marginBottom: 48, color: '#0f172a', letterSpacing: '-0.02em' }}>Everything you need to respond faster</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(290px, 1fr))', gap: 20 }}>
                    {features.map((f, i) => (
                        <div key={i} className="glass-panel" style={{ padding: '1.5rem', borderRadius: 20, background: 'rgba(255, 255, 255, 0.15)', backdropFilter: 'blur(32px) saturate(120%)', WebkitBackdropFilter: 'blur(32px) saturate(120%)', border: '1px solid rgba(59, 177, 141, 0.5)' }}>
                            <div style={{ background: f.bg, width: 44, height: 44, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16 }}>
                                {f.icon}
                            </div>
                            <h3 style={{ fontSize: 17, fontWeight: 700, marginBottom: 8, color: '#0f172a' }}>{f.title}</h3>
                            <p style={{ color: '#475569', fontSize: 14, lineHeight: 1.7, fontWeight: 500 }}>{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ── About & Contact ── */}
            <section style={{ maxWidth: 1100, margin: '0 auto', padding: '0 24px 80px', position: 'relative', zIndex: 10 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 40 }}>
                    <div className="glass-panel" style={{ padding: '2.5rem', borderRadius: 20 }}>
                        <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16, color: '#0f172a' }}>About invncble</h2>
                        <p style={{ color: '#475569', fontSize: 15, lineHeight: 1.7, fontWeight: 500 }}>
                            invncble is a cutting-edge disease outbreak prediction platform designed to empower healthcare professionals and public health authorities. By leveraging advanced machine learning models and real-time geographical data, we provide early warnings and actionable insights to mitigate the impact of infectious diseases globally.
                        </p>
                    </div>
                    <div className="glass-panel" style={{ padding: '2.5rem', borderRadius: 20 }}>
                        <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16, color: '#0f172a' }}>Contact Us</h2>
                        <p style={{ color: '#475569', fontSize: 15, lineHeight: 1.7, fontWeight: 500, marginBottom: 24 }}>
                            Have questions, need support, or interested in deploying our intelligence platform at your health institution? We are here to help.
                        </p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: '#10b981', fontWeight: 600 }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
                            <a href="mailto:noreply.cyberpulse@gmail.com" style={{ textDecoration: 'none', color: 'inherit' }}>noreply.cyberpulse@gmail.com</a>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer style={{ textAlign: 'center', padding: '24px', color: '#475569', fontSize: 13, borderTop: '1px solid rgba(15, 23, 42, 0.08)', position: 'relative', zIndex: 10, fontWeight: 500 }}>
                invncble · Authorized Healthcare Personnel Only
            </footer>
        </div>
    );
}
