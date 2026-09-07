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
 * Time-series chart of how many measured features drifted in each historical check.
 * Plots both the raw count (red) and the percentage (amber, dashed).
 * Hidden when there's only zero or one history entry.
 */
export const DriftHistoryChart: React.FC<DriftHistoryChartProps> = ({ history }) => {
    if (history.length <= 1) return null;

    const data = [...history].reverse().map(h => {
        // Persisted counts also include schema changes, including legacy target
        // artifacts. Use the recorded feature evidence for both numerator and
        // denominator; an unevaluated check has no distribution verdict.
        const columns = h.summary ? Object.values(h.summary) : null;
        const drifted = columns ? columns.filter(c => c.drifted).length : null;
        return {
            date: h.created_at?.split('T')[0] ?? '',
            drifted,
            pct: columns?.length ? Math.round(((drifted ?? 0) / columns.length) * 100) : null,
        };
    });

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
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            tickLine={false}
                            axisLine={{ stroke: '#cbd5e1' }}
                        />
                        <YAxis
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            tickLine={false}
                            axisLine={false}
                        />
                        <RechartsTooltip
                            contentStyle={{
                                backgroundColor: 'hsl(var(--popover))',
                                borderColor: 'hsl(var(--border))',
                                color: 'hsl(var(--popover-foreground))',
                                borderRadius: '6px',
                            }}
                            itemStyle={{ color: 'hsl(var(--popover-foreground))' }}
                            labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
                            formatter={(value: number, name: string) => {
                                if (name === 'Drifted Features') return [value, name];
                                if (name === 'Drift %') return [`${value}%`, name];
                                return [value, name];
                            }}
                        />
                        <Legend verticalAlign="top" height={36} iconType="circle" />
                        <Line
                            type="monotone"
                            dataKey="drifted"
                            name="Drifted Features"
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
