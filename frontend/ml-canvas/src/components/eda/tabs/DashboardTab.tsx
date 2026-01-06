import React, { useMemo } from 'react';
import { OverviewCards } from '../OverviewCards';
import { AlertsSection } from '../AlertsSection';
import { Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend } from 'recharts';
import { FileText, Database, AlertTriangle, EyeOff, Eye } from 'lucide-react';

interface DashboardTabProps {
    profile: any;
    onToggleExclude?: (column: string, exclude: boolean) => void;
    excludedCols?: string[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export const DashboardTab: React.FC<DashboardTabProps> = ({ profile, onToggleExclude, excludedCols = [] }) => {
    
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
            .slice(0, 20)
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
                        <div className="h-64 overflow-y-auto pr-2 space-y-3">
                            {missingData.map((item: any, index: number) => {
                                const isExcluded = excludedCols.includes(item.name);
                                return (
                                    <div key={index} className="flex items-center gap-3 text-sm group">
                                        <div className="w-32 truncate text-gray-600 dark:text-gray-300 font-medium" title={item.name}>
                                            {item.name}
                                        </div>
                                        <div className="flex-1 h-2 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                                            <div 
                                                className="h-full rounded-full transition-all duration-500"
                                                style={{ 
                                                    width: `${item.value}%`,
                                                    backgroundColor: `hsl(10, 80%, ${60 - (index * 2)}%)`
                                                }}
                                            />
                                        </div>
                                        <div className="w-12 text-right text-gray-500 text-xs">
                                            {item.value.toFixed(1)}%
                                        </div>
                                        {onToggleExclude && (
                                            <button
                                                onClick={() => onToggleExclude(item.name, !isExcluded)}
                                                className={`p-1.5 rounded-md transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100 ${
                                                    isExcluded 
                                                    ? 'bg-red-100 text-red-600 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 opacity-100' 
                                                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 dark:hover:text-gray-300'
                                                }`}
                                                title={isExcluded ? "Include Column" : "Exclude Column"}
                                            >
                                                {isExcluded ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        )}
                                    </div>
                                );
                            })}
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
