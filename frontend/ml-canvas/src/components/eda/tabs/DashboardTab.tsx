import React, { useMemo } from 'react';
import { OverviewCards } from '../OverviewCards';
import { AlertsSection } from '../AlertsSection';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';
import { FileText, Database, AlertTriangle } from 'lucide-react';

interface DashboardTabProps {
    profile: any;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export const DashboardTab: React.FC<DashboardTabProps> = ({ profile }) => {
    
    const dataTypeData = useMemo(() => {
        if (!profile?.columns) return [];
        const counts: Record<string, number> = {};
        Object.values(profile.columns).forEach((col: any) => {
            counts[col.dtype] = (counts[col.dtype] || 0) + 1;
        });
        return Object.entries(counts).map(([name, value]) => ({ name, value }));
    }, [profile]);

    const missingData = useMemo(() => {
        if (!profile?.columns) return [];
        return Object.values(profile.columns)
            .filter((col: any) => col.missing_percentage > 0)
            .sort((a: any, b: any) => b.missing_percentage - a.missing_percentage)
            .slice(0, 10)
            .map((col: any) => ({
                name: col.name,
                value: col.missing_percentage
            }));
    }, [profile]);

    return (
        <div className="space-y-6">
            <OverviewCards profile={profile} />
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Data Types Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                        <Database className="w-5 h-5 mr-2 text-blue-500" />
                        Column Types
                    </h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={dataTypeData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {dataTypeData.map((_, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip 
                                    contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                />
                                <Legend verticalAlign="bottom" height={36}/>
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Missing Values Chart */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
                        <AlertTriangle className="w-5 h-5 mr-2 text-amber-500" />
                        Top Missing Values
                    </h3>
                    {missingData.length > 0 ? (
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={missingData}
                                    layout="vertical"
                                    margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e5e7eb" />
                                    <XAxis type="number" domain={[0, 100]} unit="%" hide />
                                    <YAxis 
                                        dataKey="name" 
                                        type="category" 
                                        width={100} 
                                        tick={{ fontSize: 12, fill: '#6b7280' }}
                                    />
                                    <Tooltip 
                                        formatter={(value: number) => [value.toFixed(1) + '%', 'Missing']}
                                        contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                        cursor={{ fill: 'transparent' }}
                                    />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                        {missingData.map((_, index) => (
                                            <Cell key={`cell-${index}`} fill={`hsl(10, 80%, ${60 - index * 5}%)`} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <div className="h-64 flex flex-col items-center justify-center text-gray-400">
                            <FileText className="w-12 h-12 mb-2 opacity-20" />
                            <p>No missing values found!</p>
                        </div>
                    )}
                </div>
            </div>

            <AlertsSection alerts={profile.alerts} />
        </div>
    );
};
