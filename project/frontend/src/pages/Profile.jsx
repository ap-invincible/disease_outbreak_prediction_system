import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { User as UserIcon, Mail, Hash, LogOut, ArrowLeft, ShieldCheck } from 'lucide-react';

export default function Profile() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    if (!user) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <p className="text-slate-500 font-medium">Please log in to view your profile.</p>
            </div>
        );
    }

    return (
        <div className="relative min-h-screen flex flex-col items-center justify-center p-6">
            <div className="bg-blobs"></div>

            <div className="relative z-10 w-full max-w-md">
                <button onClick={() => navigate(-1)} className="glass-button px-4 py-2 flex items-center gap-2 mb-6 shadow-sm">
                    <ArrowLeft className="w-4 h-4" /> Go Back
                </button>

                <div className="glass-panel p-8 text-center relative overflow-hidden shadow-xl border border-white/60">
                    <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-br from-emerald-400/20 to-sky-400/20"></div>

                    <div className="relative w-24 h-24 mx-auto bg-white rounded-full flex items-center justify-center shadow-md border-4 border-white mb-4">
                        <UserIcon className="w-12 h-12 text-slate-300" />
                        <div className="absolute bottom-0 right-0 w-6 h-6 bg-emerald-500 rounded-full border-2 border-white flex items-center justify-center">
                            <ShieldCheck className="w-3 h-3 text-white" />
                        </div>
                    </div>

                    <h1 className="text-3xl font-bold text-slate-800 mb-1 tracking-tight">{user.username}</h1>
                    <p className="text-emerald-600 font-medium text-sm mb-8 uppercase tracking-wider">Authorized Personnel</p>

                    <div className="space-y-4 text-left mb-8">
                        <div className="flex items-center gap-4 p-4 bg-white/50 backdrop-blur-md rounded-2xl border border-white/80 shadow-sm transition-transform hover:-translate-y-1">
                            <div className="w-12 h-12 rounded-xl bg-sky-500/10 flex items-center justify-center">
                                <Hash className="w-6 h-6 text-sky-500" />
                            </div>
                            <div className="flex-1">
                                <p className="text-xs text-slate-500 uppercase tracking-wider font-bold mb-1">System ID</p>
                                <p className="font-mono text-slate-700 font-semibold">{user.id || 'N/A'}</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4 p-4 bg-white/50 backdrop-blur-md rounded-2xl border border-white/80 shadow-sm transition-transform hover:-translate-y-1">
                            <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center">
                                <Mail className="w-6 h-6 text-emerald-500" />
                            </div>
                            <div className="flex-1 overflow-hidden">
                                <p className="text-xs text-slate-500 uppercase tracking-wider font-bold mb-1">Registered Email</p>
                                <p className="text-slate-700 font-semibold truncate">{user.email || 'No email registered'}</p>
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={handleLogout}
                        className="w-full flex items-center justify-center gap-2 py-3.5 rounded-xl bg-red-50 text-red-600 font-bold border border-red-100 hover:bg-red-100 hover:border-red-200 transition-all shadow-sm"
                    >
                        <LogOut className="w-5 h-5" />
                        Sign Out
                    </button>
                </div>
            </div>
        </div>
    );
}
