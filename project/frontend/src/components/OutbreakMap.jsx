import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet's default icon path issues in bundlers
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: markerIcon2x,
    iconUrl: markerIcon,
    shadowUrl: markerShadow,
});

export default function OutbreakMap({ reports }) {
    const mapContainerRef = useRef(null);
    const mapInstance = useRef(null);
    const markersLayer = useRef(null);

    useEffect(() => {
        // Only initialize the map once
        if (!mapInstance.current && mapContainerRef.current) {
            mapInstance.current = L.map(mapContainerRef.current, {
                center: [20, 0], // Default center
                zoom: 2,
                zoomControl: false, // We'll manage zoom manually or keep it clean
            });

            // Standard OpenStreetMap tiles (bright, readable)
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                maxZoom: 19,
            }).addTo(mapInstance.current);

            markersLayer.current = L.layerGroup().addTo(mapInstance.current);

            // Force recalculation after mount
            setTimeout(() => {
                mapInstance.current.invalidateSize();
            }, 100);
        }

        // Cleanup on unmount
        return () => {
            if (mapInstance.current) {
                mapInstance.current.remove();
                mapInstance.current = null;
            }
        };
    }, []);

    // Effect to update markers when reports change
    useEffect(() => {
        if (!mapInstance.current || !markersLayer.current || !reports) return;

        // Clear existing markers
        markersLayer.current.clearLayers();

        const lats = [];
        const lngs = [];

        reports.forEach(report => {
            if (!report.lat || !report.lng || isNaN(report.lat) || isNaN(report.lng)) return;

            let color = '#22c55e'; // Green (Low)
            if (report.riskLevel === 'High') color = '#ef4444'; // Red
            if (report.riskLevel === 'Medium') color = '#eab308'; // Yellow

            // Radius scales with vulnerability
            const radius = Math.max(50000, report.vulnerabilityScore * 1500);

            // Add circle to map
            const circle = L.circle([report.lat, report.lng], {
                color: color,
                fillColor: color,
                fillOpacity: 0.4,
                radius: radius,
            });

            // Add popup
            circle.bindPopup(`
                <div style="text-align: center; font-family: sans-serif;">
                    <strong style="font-size: 16px;">${report.disease}</strong><br/>
                    <span style="color: #666;">${report.locationName}</span><br/><br/>
                    <b>Risk:</b> <span style="color: ${color};">${report.riskLevel}</span><br/>
                    <b>Vulnerability:</b> ${report.vulnerabilityScore}/100
                </div>
            `);

            circle.addTo(markersLayer.current);

            lats.push(report.lat);
            lngs.push(report.lng);
        });

        // Fit bounds to show all markers automatically
        if (lats.length > 0 && lngs.length > 0) {
            const minLat = Math.min(...lats);
            const maxLat = Math.max(...lats);
            const minLng = Math.min(...lngs);
            const maxLng = Math.max(...lngs);

            // Add padding so circles aren't cut off at the edges
            mapInstance.current.fitBounds([
                [minLat - 2, minLng - 2],
                [maxLat + 2, maxLng + 2]
            ]);
        }
    }, [reports]);

    return (
        <div style={{ height: '100%', width: '100%', minHeight: '500px', borderRadius: '0 0 24px 24px', overflow: 'hidden' }}>
            <div ref={mapContainerRef} style={{ height: '100%', width: '100%' }} />
        </div>
    );
}
