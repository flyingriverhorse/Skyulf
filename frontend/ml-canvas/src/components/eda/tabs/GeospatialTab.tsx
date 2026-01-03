import React from 'react';
import { Map, Info } from 'lucide-react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { InfoTooltip } from '../../ui/InfoTooltip';
import { COLORS } from '../constants';

interface GeospatialTabProps {
    profile: any;
}

export const GeospatialTab: React.FC<GeospatialTabProps> = ({ profile }) => {
    return (
        <div className="mt-4 bg-white dark:bg-gray-800 p-6 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
            <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center">
                    <Map className="w-5 h-5 mr-2 text-blue-500" />
                    Geospatial Analysis
                    <InfoTooltip text="Visualizes data points on a map based on detected Latitude/Longitude columns. Colors indicate target values." />
                </h2>
                <div className="text-sm text-gray-500">
                    Detected columns: <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lat_col}</span>, <span className="font-mono bg-gray-100 dark:bg-gray-900 px-1 rounded">{profile.geospatial.lon_col}</span>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 h-96 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 relative z-0">
                    {(() => {
                        // Calculate bounds for the map
                        const bounds: [[number, number], [number, number]] = [
                            [profile.geospatial.min_lat, profile.geospatial.min_lon],
                            [profile.geospatial.max_lat, profile.geospatial.max_lon]
                        ];
                        
                        // Helper to get color
                        const uniqueLabels = Array.from(new Set(profile.geospatial.sample_points.map((p: any) => p.label))).filter(Boolean);
                        const isChaotic = uniqueLabels.length > 20;
                        
                        const getColor = (label: string | null) => {
                            if (!label) return '#3b82f6'; // Default blue
                            if (isChaotic) return '#3b82f6'; // Single color for chaotic
                            const index = uniqueLabels.indexOf(label);
                            return COLORS[index % COLORS.length];
                        };

                        return (
                            <MapContainer 
                                bounds={bounds} 
                                style={{ height: '100%', width: '100%' }}
                                scrollWheelZoom={true}
                            >
                                <TileLayer
                                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                />
                                {profile.geospatial.sample_points.map((point: any, idx: number) => (
                                    <CircleMarker 
                                        key={idx} 
                                        center={[point.lat, point.lon]} 
                                        radius={5}
                                        pathOptions={{ 
                                            color: getColor(point.label), 
                                            fillColor: getColor(point.label), 
                                            fillOpacity: 0.7,
                                            weight: 1
                                        }}
                                    >
                                        <Popup>
                                            <div className="text-xs">
                                                <strong>Lat:</strong> {point.lat.toFixed(4)}<br/>
                                                <strong>Lon:</strong> {point.lon.toFixed(4)}<br/>
                                                {point.label && <><strong>{profile.target_col}:</strong> {point.label}</>}
                                            </div>
                                        </Popup>
                                    </CircleMarker>
                                ))}
                            </MapContainer>
                        );
                    })()}
                </div>

                <div className="space-y-4">
                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                        <h3 className="text-sm font-medium text-gray-500 mb-3">Spatial Bounds</h3>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                                <span className="block text-xs text-gray-400">Latitude Range</span>
                                <span className="font-mono">{profile.geospatial.min_lat.toFixed(4)} to {profile.geospatial.max_lat.toFixed(4)}</span>
                            </div>
                            <div>
                                <span className="block text-xs text-gray-400">Longitude Range</span>
                                <span className="font-mono">{profile.geospatial.min_lon.toFixed(4)} to {profile.geospatial.max_lon.toFixed(4)}</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                        <h3 className="text-sm font-medium text-gray-500 mb-3">Centroid</h3>
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full bg-red-500"></div>
                            <span className="font-mono text-sm">
                                {profile.geospatial.centroid_lat.toFixed(4)}, {profile.geospatial.centroid_lon.toFixed(4)}
                            </span>
                        </div>
                    </div>
                    
                    <div className="flex items-start gap-2 bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                        <Info className="w-4 h-4 text-blue-500 mt-0.5 shrink-0" />
                        <div className="text-sm text-blue-700 dark:text-blue-300">
                            <p>
                                This scatter plot shows the spatial distribution of your data.
                                {profile.target_col && " Points are colored by the target variable."}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
