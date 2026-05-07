import React from 'react';
import {
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip as RechartsTooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import type { DriftHistoryEntry } from '../../core/api/monitoring';

interface DriftHistoryChartProps {
    history: DriftHistoryEntry[];
}

/**
 * Time-series chart of how many columns drifted in each historical check.
 * Plots both the raw count (red) and the percentage (amber, dashed).
 * Hidden when there's only zero or one history entry.
 */
export const DriftHistoryChart: React.FC<DriftHistoryChartProps> = ({ history }) => {
    if (history.length <= 1) return null;

    const data = [...history].reverse().map(h => ({
        date: h.created_at?.split('T')[0] ?? '',
        drifted: h.drifted_columns_count ?? 0,
        total: h.total_columns ?? 0,
        pct: h.total_columns
            ? Math.round(((h.drifted_columns_count ?? 0) / h.total_columns) * 100)
            : 0,
    }));

    return (
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700 mt-6">
            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <TrendingUp size={18} /> Drift History
                <span className="text-xs font-normal text-gray-400 ml-1">({history.length} checks)</span>
            </h2>
            <div className="h-[280px]">
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                        <CartesianGrid
                            strokeDasharray="3 3"
                            className="stroke-gray-200 dark:stroke-slate-700"
                            vertical={false}
                        />
                        <XAxis
                            dataKey="date"
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: '#cbd5e1' }}
                        />
                        <YAxis
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                        />
                        <RechartsTooltip
                            contentStyle={{
                                backgroundColor: '#1e293b',
                                borderColor: '#334155',
                                color: '#f8fafc',
                                borderRadius: '6px',
                            }}
                            itemStyle={{ color: '#f8fafc' }}
                            labelStyle={{ color: '#94a3b8' }}
                            formatter={(value: number, name: string) => {
                                if (name === 'Drifted Columns') return [value, name];
                                if (name === 'Drift %') return [`${value}%`, name];
                                return [value, name];
                            }}
                        />
                        <Legend verticalAlign="top" height={36} iconType="circle" />
                        <Line
                            type="monotone"
                            dataKey="drifted"
                            name="Drifted Columns"
                            stroke="#ef4444"
                            strokeWidth={2}
                            dot={{ r: 3, fill: '#ef4444' }}
                        />
                        <Line
                            type="monotone"
                            dataKey="pct"
                            name="Drift %"
                            stroke="#f59e0b"
                            strokeWidth={2}
                            dot={{ r: 3, fill: '#f59e0b' }}
                            strokeDasharray="5 5"
                        />
                    </LineChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};
