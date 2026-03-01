import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { ShieldAlert, Activity, TrendingUp, Clock, AlertTriangle, ArrowLeft } from 'lucide-react';
import OutbreakMap from '../components/OutbreakMap';

export default function Dashboard() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [data, setData] = useState({ reports: [], total_cases_reported: 0, active_high_risk_outbreaks: 0 });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await axios.get('http://localhost:5000/api/dashboard/data', {
                    headers: { Authorization: `Bearer ${user.token}` }
                });
                setData(res.data);
            } catch (error) {
                console.error("Dashboard fetch error:", error);
            }
            setLoading(false);
        };
        fetchData();
    }, [user]);

    if (loading) return <div className="min-h-screen flex items-center justify-center text-xl">Loading Analytics...</div>;

    return (
        <div className="relative min-h-screen flex flex-col">
            <div className="bg-blobs"></div>
            <div className="p-6 md:p-8 flex-1 flex flex-col relative z-10 space-y-8">
                <header className="flex justify-between items-center bg-white/40 backdrop-blur-md p-4 px-6 rounded-2xl border border-white/80 shadow-sm">
                    <h1 className="text-3xl font-bold flex items-center gap-3 text-slate-800 tracking-tight">
                        <ShieldAlert className="text-emerald-500 w-8 h-8" />
                        Primary Outbreak Dashboard
                    </h1>
                    <button onClick={() => navigate('/')} className="glass-button px-4 py-2 flex items-center gap-2">
                        <ArrowLeft className="w-4 h-4" /> Home
                    </button>
                </header>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div className="glass-panel p-6 flex items-center gap-4">
                        <div className="p-4 bg-emerald-500/10 rounded-2xl border border-emerald-500/20"><Activity className="w-8 h-8 text-emerald-500" /></div>
                        <div>
                            <div className="text-sm text-slate-500 font-semibold uppercase tracking-wider">Total Tracked</div>
                            <div className="text-3xl font-bold text-slate-800">{data.total_cases_reported}</div>
                        </div>
                    </div>
                    <div className="glass-panel p-6 flex items-center gap-4 border border-rose-400/40">
                        <div className="p-4 bg-rose-500/10 rounded-2xl border border-rose-500/20"><AlertTriangle className="w-8 h-8 text-rose-500" /></div>
                        <div>
                            <div className="text-sm text-slate-500 font-semibold uppercase tracking-wider">High Risk Epicenters</div>
                            <div className="text-3xl font-bold text-rose-600">{data.active_high_risk_outbreaks}</div>
                        </div>
                    </div>
                </div>

                <div className="grid lg:grid-cols-3 gap-8 flex-1">

                    {/* Analytics Feed */}
                    <div className="lg:col-span-1 glass-panel flex flex-col overflow-hidden max-h-[800px]">
                        <div className="p-6 border-b border-slate-200 bg-white/40">
                            <h2 className="text-xl font-bold text-slate-800">Latest Active Detections</h2>
                        </div>
                        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                            {data.reports.length === 0 ? (
                                <div className="text-slate-500 text-center py-10 font-medium">No outbreak logic processed yet.</div>
                            ) : (
                                data.reports.slice().reverse().map((report) => (
                                    <div key={report.id} className="bg-white/60 p-5 rounded-2xl border border-emerald-500/30 hover:border-emerald-500/50 hover:bg-white hover:shadow-md transition-all">
                                        <div className="flex justify-between items-start mb-4">
                                            <div>
                                                <h3 className="font-bold text-lg text-slate-800">{report.disease}</h3>
                                                <div className="text-sm text-slate-500 font-medium">{report.locationName}</div>
                                            </div>
                                            <span className={`px-3 py-1 text-xs font-bold rounded-full uppercase ${report.riskLevel === 'High' ? 'bg-red-100 text-red-700 border border-red-200' :
                                                report.riskLevel === 'Medium' ? 'bg-amber-100 text-amber-700 border border-amber-200' :
                                                    'bg-emerald-100 text-emerald-700 border border-emerald-200'
                                                }`}>
                                                {report.riskLevel} Risk
                                            </span>
                                        </div>

                                        <div className="grid grid-cols-2 gap-3">
                                            <div className="bg-slate-50/80 p-3 rounded-xl border border-slate-100">
                                                <div className="text-xs text-slate-500 font-semibold flex flex-col gap-1 items-start uppercase">
                                                    <TrendingUp className="w-3 h-3" /> Growth Rate
                                                </div>
                                                <div className="font-bold text-slate-700 text-lg">{report.growthRate}x</div>
                                            </div>
                                            <div className="bg-slate-50/80 p-3 rounded-xl border border-slate-100">
                                                <div className="text-xs text-slate-500 font-semibold flex flex-col gap-1 items-start uppercase">
                                                    <Clock className="w-3 h-3" /> Doubling Time
                                                </div>
                                                <div className="font-bold text-slate-700 text-lg">{report.doublingTime}d</div>
                                            </div>
                                        </div>

                                        <div className="mt-4 flex items-center gap-3">
                                            <span className="text-xs text-slate-500 font-semibold uppercase">Vulnerability:</span>
                                            <div className="flex-1 h-2 bg-slate-200 rounded-full overflow-hidden">
                                                <div className="h-full bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500" style={{ width: `${report.vulnerabilityScore}%` }} />
                                            </div>
                                            <span className="text-xs font-bold text-slate-700">{report.vulnerabilityScore}</span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Tactical Map */}
                    <div className="lg:col-span-2 glass-panel p-2 flex flex-col">
                        <div className="px-6 py-4 border-b border-slate-200 flex justify-between items-center bg-white/60 backdrop-blur-md rounded-t-[1.3rem]">
                            <h2 className="text-xl font-bold text-slate-800">Geographical Heatmap</h2>
                            <div className="flex items-center gap-4 text-xs font-bold text-slate-600">
                                <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-rose-500"></div> High Risk</span>
                                <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-amber-500"></div> Medium Risk</span>
                                <span className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-emerald-500"></div> Low Risk</span>
                            </div>
                        </div>
                        <div className="flex-1 bg-white/40 rounded-b-[1.3rem] relative overflow-hidden border-t border-white shadow-inner">
                            <OutbreakMap reports={data.reports} />
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}
