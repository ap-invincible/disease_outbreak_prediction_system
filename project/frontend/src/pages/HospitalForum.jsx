import React, { useState, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { Activity, MapPin, Thermometer, ShieldAlert, LogOut } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet default marker icon issue in bundlers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: markerIcon2x,
    iconUrl: markerIcon,
    shadowUrl: markerShadow,
});

export default function HospitalForum() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const [formData, setFormData] = useState({
        areaRegion: '',
        city: '',
        latitude: '',
        longitude: '',
        fever: 98.6,
        body_pain: 0,
        runny_nose: 0,
        headache: 0,
        fatigue: 0,
        vomiting_diarrhea: 0
    });

    const [mapError, setMapError] = useState('');
    const [showMap, setShowMap] = useState(false);
    const [locating, setLocating] = useState(false);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Vanilla Leaflet refs
    const mapContainerRef = useRef(null);
    const leafletMap = useRef(null);
    const leafletMarker = useRef(null);

    const initOrUpdateMap = useCallback((lat, lng) => {
        // If map doesn't exist yet, create it
        if (!leafletMap.current && mapContainerRef.current) {
            leafletMap.current = L.map(mapContainerRef.current, {
                center: [lat, lng],
                zoom: 14,
                zoomControl: true,
            });

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
                maxZoom: 19,
            }).addTo(leafletMap.current);

            // Create draggable marker
            leafletMarker.current = L.marker([lat, lng], { draggable: true }).addTo(leafletMap.current);
            leafletMarker.current.bindPopup('Drag to adjust location').openPopup();

            leafletMarker.current.on('dragend', function () {
                const pos = leafletMarker.current.getLatLng();
                setFormData(prev => ({ ...prev, latitude: pos.lat, longitude: pos.lng }));
            });

            // Also allow clicking to reposition
            leafletMap.current.on('click', function (e) {
                leafletMarker.current.setLatLng(e.latlng);
                setFormData(prev => ({ ...prev, latitude: e.latlng.lat, longitude: e.latlng.lng }));
            });

            // Force Leaflet to recalculate sizes after render
            setTimeout(() => {
                leafletMap.current.invalidateSize();
            }, 100);
        } else if (leafletMap.current) {
            // Map already exists, just move it
            leafletMap.current.setView([lat, lng], 14);
            leafletMarker.current.setLatLng([lat, lng]);
            leafletMap.current.invalidateSize();
        }
    }, []);

    const handleLocateOnMap = async () => {
        setMapError('');
        if (!formData.areaRegion || !formData.city) {
            setMapError('Please enter both Area/Region and City Name first.');
            return;
        }

        setLocating(true);
        const address = `${formData.areaRegion}, ${formData.city}`;

        try {
            // Free OpenStreetMap Nominatim Geocoding API (no key needed!)
            const response = await axios.get(
                'https://nominatim.openstreetmap.org/search',
                {
                    params: {
                        format: 'json',
                        q: address,
                        limit: 1
                    }
                }
            );

            if (response.data && response.data.length > 0) {
                const lat = parseFloat(response.data[0].lat);
                const lng = parseFloat(response.data[0].lon);
                setFormData(prev => ({ ...prev, latitude: lat, longitude: lng }));
                setShowMap(true);

                // Use requestAnimationFrame to ensure DOM is painted before initializing map
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        initOrUpdateMap(lat, lng);
                    });
                });
            } else {
                setMapError('Could not locate this region. Please try a different name.');
            }
        } catch (err) {
            console.error("Geocoding Error:", err);
            setMapError('Geocoding service error. Please try again.');
        }
        setLocating(false);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const locationName = `${formData.areaRegion}, ${formData.city}`.trim();
            const res = await axios.post('http://localhost:5000/api/hospital/submit', {
                ...formData,
                locationName: locationName,
                latitude: parseFloat(formData.latitude) || 0,
                longitude: parseFloat(formData.longitude) || 0,
                fever: parseFloat(formData.fever)
            }, {
                headers: { Authorization: `Bearer ${user.token}` }
            });

            setResult(res.data.pipeline_results);
        } catch (error) {
            console.error(error);
            alert('Failed to submit report');
        }
        setLoading(false);
    };

    return (
        <div className="relative min-h-screen flex flex-col p-6 md:p-12 z-10">
            <div className="bg-blobs"></div>
            <header className="flex justify-between items-center mb-12 max-w-5xl mx-auto w-full bg-white/40 backdrop-blur-md p-4 px-6 rounded-2xl border border-white/80 shadow-sm">
                <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-800 tracking-tight">
                    <Activity className="text-emerald-500 w-8 h-8" />
                    Hospital Forum Intake
                </h1>
                <div className="flex gap-4">
                    <button onClick={() => navigate('/dashboard')} className="glass-button px-4 py-2 text-sm flex items-center gap-2">
                        <ShieldAlert className="w-4 h-4" /> View Dashboard
                    </button>
                    <button onClick={() => { logout(); navigate('/'); }} className="glass-button px-4 py-2 text-sm text-rose-600 hover:bg-rose-50 border-rose-200 flex items-center gap-2 font-semibold">
                        <LogOut className="w-4 h-4" /> Logout
                    </button>
                </div>
            </header>

            <div className="max-w-5xl mx-auto grid md:grid-cols-2 gap-8 w-full">
                <div className="glass-panel p-8">
                    <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-4">New Patient Log</h2>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div className="space-y-4">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                <MapPin className="w-4 h-4" /> Location Data
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                <input
                                    type="text" required placeholder="Area / Region (e.g. Sector-23B)"
                                    className="glass-input w-full"
                                    value={formData.areaRegion} onChange={e => setFormData({ ...formData, areaRegion: e.target.value })}
                                />
                                <input
                                    type="text" required placeholder="City (e.g. Gurugram)"
                                    className="glass-input w-full"
                                    value={formData.city} onChange={e => setFormData({ ...formData, city: e.target.value })}
                                />
                            </div>

                            <button
                                type="button"
                                onClick={handleLocateOnMap}
                                disabled={locating}
                                className="w-full glass-button bg-sky-50 hover:bg-sky-100 py-2 text-sm text-sky-700 flex items-center justify-center gap-2 font-semibold"
                            >
                                <MapPin className="w-4 h-4" />
                                {locating ? 'Locating...' : 'Locate on Map'}
                            </button>

                            {mapError && <div className="text-rose-500 text-sm font-medium">{mapError}</div>}

                            {showMap && (
                                <div className="space-y-2">
                                    <div
                                        ref={mapContainerRef}
                                        style={{ height: '250px', width: '100%', borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(0,0,0,0.1)' }}
                                    />
                                    <div className="text-xs text-slate-500 font-medium text-center">
                                        Click or drag the marker to adjust the precise location
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="space-y-4 pt-4 border-t border-slate-200">
                            <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                                <Thermometer className="w-4 h-4" /> Clinical Symptoms
                            </h3>

                            <div>
                                <label className="block text-xs font-bold text-slate-600 uppercase tracking-wider mb-2">Fever (°F)</label>
                                <input
                                    type="number" step="0.1" required
                                    className="glass-input w-full"
                                    value={formData.fever} onChange={e => setFormData({ ...formData, fever: e.target.value })}
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                {['body_pain', 'runny_nose', 'headache', 'fatigue', 'vomiting_diarrhea'].map(sym => (
                                    <label key={sym} className="flex items-center gap-3 p-3 rounded-xl border border-white bg-white/40 hover:bg-white/80 cursor-pointer transition-all shadow-sm">
                                        <input
                                            type="checkbox"
                                            className="w-5 h-5 rounded border-slate-300 text-emerald-500 focus:ring-emerald-500"
                                            checked={formData[sym] === 1}
                                            onChange={e => setFormData({ ...formData, [sym]: e.target.checked ? 1 : 0 })}
                                        />
                                        <span className="capitalize text-slate-700 font-medium text-sm">{sym.replace('_', ' ')}</span>
                                    </label>
                                ))}
                            </div>
                        </div>

                        <button type="submit" disabled={loading} className="w-full glass-button bg-gradient-to-r from-emerald-500 to-sky-500 py-4 font-bold text-lg text-white shadow-md hover:opacity-90 border-transparent">
                            {loading ? 'Processing through Pipeline...' : 'Submit & Analyze'}
                        </button>
                    </form>
                </div>

                <div>
                    {result ? (
                        <div className="glass-panel p-8 border-l-[6px] border-emerald-400 animate-fade-in-down h-full">
                            <h2 className="text-xl font-bold text-slate-800 mb-6 border-b border-slate-200 pb-4">Pipeline Results</h2>

                            <div className="space-y-6">
                                {result.disease === 'Healthy' ? (
                                    <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-6 text-center">
                                        <ShieldAlert className="w-10 h-10 text-emerald-500 mx-auto mb-3" />
                                        <div className="text-2xl font-bold text-emerald-700 mb-2">No Disease Detected</div>
                                        <div className="text-sm text-emerald-600 font-medium">
                                            The patient's vitals are within normal range. No fever or clinical symptoms were reported. No further analysis required.
                                        </div>
                                    </div>
                                ) : (
                                    <>
                                        <div>
                                            <div className="text-xs font-bold text-slate-500 uppercase mb-1">Detected Disease</div>
                                            <div className="text-3xl font-bold text-emerald-600">{result.disease}</div>
                                        </div>

                                        {result.risk_level !== 'None' ? (
                                            <>
                                                <div className="grid grid-cols-2 gap-4">
                                                    <div className="bg-slate-50/80 p-4 rounded-2xl border border-slate-200">
                                                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Growth Rate</div>
                                                        <div className="text-xl font-bold text-slate-800">{result.growth_rate}x</div>
                                                    </div>
                                                    <div className="bg-slate-50/80 p-4 rounded-2xl border border-slate-200">
                                                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Doubling Time</div>
                                                        <div className="text-xl font-bold text-slate-800">{result.doubling_time} days</div>
                                                    </div>
                                                </div>

                                                <div className="bg-slate-50/80 p-4 rounded-2xl border border-slate-200">
                                                    <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">Geographic Vulnerability</div>
                                                    <div className="flex items-center gap-3">
                                                        <div className="flex-1 h-3 bg-slate-200 rounded-full overflow-hidden">
                                                            <div
                                                                className={`h-full rounded-full ${result.vulnerability > 70 ? 'bg-rose-500' : result.vulnerability > 40 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                                                                style={{ width: `${result.vulnerability}%` }}
                                                            />
                                                        </div>
                                                        <span className="font-bold text-slate-700">{result.vulnerability}/100</span>
                                                    </div>
                                                </div>

                                                <div className={`p-4 rounded-2xl border ${result.risk_level === 'High' ? 'bg-rose-50 border-rose-200 text-rose-800' :
                                                    result.risk_level === 'Medium' ? 'bg-amber-50 border-amber-200 text-amber-800' :
                                                        'bg-emerald-50 border-emerald-200 text-emerald-800'
                                                    }`}>
                                                    <div className="text-xs font-bold opacity-80 mb-1 uppercase tracking-wider">48-Hour Outbreak Risk</div>
                                                    <div className="text-3xl font-black uppercase">{result.risk_level}</div>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="bg-emerald-50 border border-emerald-200 rounded-2xl p-6 text-center">
                                                <ShieldAlert className="w-8 h-8 text-emerald-500 mx-auto mb-3" />
                                                <div className="text-emerald-700 font-bold text-lg mb-1">No Outbreak Detected</div>
                                                <div className="text-sm text-emerald-600 font-medium">
                                                    Disease identified, but local case count ({result.case_count || 1}) is below the threshold required to signal an active outbreak trend.
                                                </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="glass-panel p-8 h-full flex flex-col items-center justify-center text-center border-dashed border-2 border-slate-300">
                            <Activity className="w-16 h-16 mb-4 text-emerald-400 opacity-80" />
                            <p className="text-slate-500 font-medium max-w-xs">Submit patient data to run the multi-model prediction pipeline.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
