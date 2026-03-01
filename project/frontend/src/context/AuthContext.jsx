import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export function useAuth() {
    return useContext(AuthContext);
}

export function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('token');
        const username = localStorage.getItem('username');
        const email = localStorage.getItem('email');
        const id = localStorage.getItem('id');
        if (token && username) {
            setUser({ token, username, email, id });
        }
        setLoading(false);
    }, []);

    const login = async (username, password) => {
        try {
            const res = await axios.post('http://localhost:5000/api/auth/login', { username, password });
            localStorage.setItem('token', res.data.token);
            localStorage.setItem('username', res.data.username);
            if (res.data.email) localStorage.setItem('email', res.data.email);
            if (res.data.id) localStorage.setItem('id', res.data.id);
            setUser({ token: res.data.token, username: res.data.username, email: res.data.email, id: res.data.id });
            return { success: true };
        } catch (error) {
            return { success: false, error: error.response?.data?.error || 'Login failed' };
        }
    };

    const register = async (username, email, password) => {
        try {
            await axios.post('http://localhost:5000/api/auth/register', { username, email, password });
            return await login(username, password);
        } catch (error) {
            return { success: false, error: error.response?.data?.error || 'Registration failed' };
        }
    };

    const logout = () => {
        localStorage.removeItem('token');
        localStorage.removeItem('username');
        localStorage.removeItem('email');
        localStorage.removeItem('id');
        setUser(null);
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading }}>
            {!loading && children}
        </AuthContext.Provider>
    );
}
