import React, { useState } from 'react';
import {
    ArrowDown,
    ArrowUp,
    ArrowUpDown,
    BarChart2,
    CheckCircle,
    ChevronDown,
    ChevronUp,
    Info,
    Lightbulb,
    Shield,
    ShieldAlert,
    TrendingUp,
    XCircle,
} from 'lucide-react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip as RechartsTooltip,
    XAxis,
    YAxis,
} from 'recharts';
import type { ColumnDrift, DriftReport } from '../../core/api/monitoring';
import { MetricTooltip } from './MetricTooltip';
import { Sparkline } from './Sparkline';
import type { SortConfig } from './_hooks/useSortConfig';

interface DriftTableProps {
    report: DriftReport;
    showOnlyDrifted: boolean;
    sortConfig: SortConfig | null;
    onSort: (key: string) => void;
    columnSparklines: Record<string, number[]>;
}

const METRIC_KEY_MAP: Record<string, string> = {
    wasserstein: 'wasserstein_distance',
    psi: 'psi',
    kl: 'kl_divergence',
    ks: 'ks_test_p_value',
};

const sortRows = (
    rows: ColumnDrift[],
    sortConfig: SortConfig,
    fi: Record<string, number> | undefined,
): ColumnDrift[] => {
    const getMetric = (col: ColumnDrift, metric: string) =>
        col.metrics.find(m => m.metric === metric)?.value ?? 0;
    return [...rows].sort((a, b) => {
        let cmp = 0;
        switch (sortConfig.key) {
            case 'column':
                cmp = a.column.localeCompare(b.column);
                break;
            case 'status':
                cmp = (a.drift_detected ? 1 : 0) - (b.drift_detected ? 1 : 0);
                break;
            case 'risk':
                cmp = (fi?.[a.column] ?? 0) - (fi?.[b.column] ?? 0);
                break;
            default: {
                const metric = METRIC_KEY_MAP[sortConfig.key];
                if (metric) cmp = getMetric(a, metric) - getMetric(b, metric);
            }
        }
        return sortConfig.dir === 'asc' ? cmp : -cmp;
    });
};

/** Renders the sort indicator icon for a column header. */
const SortIcon: React.FC<{ active: boolean; dir: 'asc' | 'desc' | undefined }> = ({ active, dir }) => {
    if (!active) return <ArrowUpDown size={11} className="opacity-30" />;
    return dir === 'asc' ? <ArrowUp size={11} /> : <ArrowDown size={11} />;
};

/** Distribution histogram inside an expanded row. */
const DistributionChart: React.FC<{ distribution: NonNullable<ColumnDrift['distribution']> }> = ({
    distribution,
}) => (
    <div className="h-[350px] w-full bg-white dark:bg-slate-800 p-6 rounded border dark:border-slate-700 shadow-sm">
        <h4 className="text-sm font-semibold mb-6 flex items-center gap-2 text-slate-700 dark:text-slate-300">
            <BarChart2 size={16} /> Distribution Comparison
        </h4>
        <ResponsiveContainer width="100%" height="85%">
            <BarChart data={distribution.bins} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-slate-700" vertical={false} />
                <XAxis
                    dataKey={(bin: { bin_start: number }) => `${bin.bin_start.toFixed(2)}`}
                    className="text-xs"
                    tick={{ fill: '#64748b' }}
                    tickLine={false}
                    axisLine={{ stroke: '#cbd5e1' }}
                />
                <YAxis className="text-xs" tick={{ fill: '#64748b' }} tickLine={false} axisLine={false} />
                <RechartsTooltip
                    cursor={{ fill: 'rgba(0,0,0,0.05)' }}
                    contentStyle={{
                        backgroundColor: '#1e293b',
                        borderColor: '#334155',
                        color: '#f8fafc',
                        borderRadius: '6px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    }}
                    itemStyle={{ color: '#f8fafc' }}
                    labelStyle={{ color: '#94a3b8', marginBottom: '0.5rem' }}
                />
                <Legend verticalAlign="top" height={36} iconType="circle" />
                <Bar
                    dataKey="reference_count"
                    name="Reference (Training)"
                    fill="#94a3b8"
                    radius={[4, 4, 0, 0]}
                    barSize={30}
                />
                <Bar
                    dataKey="current_count"
                    name="Current (Production)"
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                    barSize={30}
                />
            </BarChart>
        </ResponsiveContainer>
    </div>
);

/** Risk badge based on drift × importance rank. */
const RiskBadge: React.FC<{
    importance: number | undefined;
    rank: number | null;
    drifted: boolean;
    maxImportance: number;
}> = ({ importance, rank, drifted, maxImportance }) => {
    if (importance == null) return <span className="text-xs text-gray-400">—</span>;
    const isHigh = drifted && rank != null && rank <= 5;
    const isMedium = drifted && rank != null && rank <= 15 && !isHigh;
    return (
        <div className="flex items-center gap-1.5">
            {isHigh ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300">
                    <ShieldAlert size={12} /> High
                </span>
            ) : isMedium ? (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300">
                    <ShieldAlert size={12} /> Medium
                </span>
            ) : (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400">
                    <Shield size={12} /> Low
                </span>
            )}
            <div
                className="w-12 h-1.5 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden"
                title={`Importance: ${((importance * 100) / maxImportance).toFixed(0)}%`}
            >
                <div
                    className="h-full bg-indigo-500 rounded-full"
                    style={{ width: `${maxImportance > 0 ? (importance / maxImportance) * 100 : 0}%` }}
                />
            </div>
        </div>
    );
};

export const DriftTable: React.FC<DriftTableProps> = ({
    report,
    showOnlyDrifted,
    sortConfig,
    onSort,
    columnSparklines,
}) => {
    const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});

    // Collapse all rows whenever a fresh analysis lands (matches the original
    // monolith's `setExpandedRows({})` inside `handleCalculate`).
    React.useEffect(() => {
        setExpandedRows({});
    }, [report]);

    // Pressing Esc collapses every expanded row.
    React.useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setExpandedRows({});
        };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, []);

    const toggleRow = (col: string) =>
        setExpandedRows(prev => ({ ...prev, [col]: !prev[col] }));

    const fi = report.feature_importances;
    const maxImportance = fi ? Math.max(...Object.values(fi)) : 0;
    const hasSparklines = Object.keys(columnSparklines).length > 0;

    let rows: ColumnDrift[] = Object.values(report.column_drifts);
    if (showOnlyDrifted) rows = rows.filter(c => c.drift_detected);
    if (sortConfig) rows = sortRows(rows, sortConfig, fi);

    const colSpan = (fi ? 8 : 7) + (hasSparklines ? 1 : 0);

    const sortHeader = (key: string, label: React.ReactNode) => (
        <th
            className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300"
            onClick={() => onSort(key)}
        >
            <span className="flex items-center gap-1">
                {label}
                <SortIcon active={sortConfig?.key === key} dir={sortConfig?.dir} />
            </span>
        </th>
    );

    return (
        <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
                <thead className="bg-gray-50 dark:bg-slate-900">
                    <tr>
                        {sortHeader('column', 'Column')}
                        {sortHeader('status', 'Status')}
                        {sortHeader(
                            'wasserstein',
                            <MetricTooltip
                                label="Wasserstein"
                                tooltip="Measures distance between distributions. Lower is better. < 0.1 usually means stable."
                                icon={<Info className="w-3 h-3 text-slate-400" />}
                            />,
                        )}
                        {sortHeader(
                            'psi',
                            <MetricTooltip
                                label="PSI"
                                tooltip="Population Stability Index. < 0.1: Stable, < 0.2: Minor Drift, > 0.2: Significant Drift."
                                icon={<Info className="w-3 h-3 text-slate-400" />}
                            />,
                        )}
                        {sortHeader(
                            'kl',
                            <MetricTooltip
                                label="KL Div"
                                tooltip="Kullback-Leibler Divergence. Measures how one probability distribution diverts from a second."
                                icon={<Info className="w-3 h-3 text-slate-400" />}
                            />,
                        )}
                        {sortHeader(
                            'ks',
                            <MetricTooltip
                                label="KS P-Value"
                                tooltip="Kolmogorov-Smirnov Test. p-value < 0.05 indicates the distributions are significantly different."
                                icon={<Info className="w-3 h-3 text-slate-400" />}
                            />,
                        )}
                        {fi &&
                            sortHeader(
                                'risk',
                                <MetricTooltip
                                    label="Risk"
                                    tooltip="Combines drift status with feature importance. High = drifted + important feature. Helps prioritize which drifts to investigate."
                                    icon={<Info className="w-3 h-3 text-slate-400" />}
                                />,
                            )}
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            Actions
                        </th>
                        {hasSparklines && (
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                <span className="flex items-center gap-1">
                                    <TrendingUp size={11} /> Trend
                                </span>
                            </th>
                        )}
                    </tr>
                </thead>
                <tbody className="bg-white dark:bg-slate-800 divide-y divide-gray-200 dark:divide-slate-700">
                    {rows.length === 0 && showOnlyDrifted ? (
                        <tr>
                            <td
                                colSpan={colSpan}
                                className="px-6 py-8 text-center text-gray-400 dark:text-slate-500 text-sm"
                            >
                                <CheckCircle size={20} className="inline mr-2 text-green-500" />
                                No drifted columns found — all features are stable.
                            </td>
                        </tr>
                    ) : (
                        rows.map(col => {
                            const wasserstein = col.metrics.find(m => m.metric === 'wasserstein_distance');
                            const psi = col.metrics.find(m => m.metric === 'psi');
                            const kl = col.metrics.find(m => m.metric === 'kl_divergence');
                            const ks = col.metrics.find(m => m.metric === 'ks_test_p_value');
                            const isExpanded = expandedRows[col.column];
                            const importance = fi?.[col.column];
                            const importanceRank = fi
                                ? Object.values(fi).filter(v => v > (importance ?? 0)).length + 1
                                : null;

                            return (
                                <React.Fragment key={col.column}>
                                    <tr className={col.drift_detected ? 'bg-red-50 dark:bg-red-900/10' : ''}>
                                        <td className="px-6 py-4 whitespace-nowrap font-medium">{col.column}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            {col.drift_detected ? (
                                                <span className="text-red-600 dark:text-red-400 flex items-center gap-1">
                                                    <XCircle size={16} /> Drifted
                                                </span>
                                            ) : (
                                                <span className="text-green-600 dark:text-green-400 flex items-center gap-1">
                                                    <CheckCircle size={16} /> Stable
                                                </span>
                                            )}
                                        </td>
                                        <td
                                            className={`px-6 py-4 whitespace-nowrap tabular-nums ${
                                                wasserstein?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''
                                            }`}
                                        >
                                            {wasserstein?.value?.toFixed(4) ?? '—'}
                                        </td>
                                        <td
                                            className={`px-6 py-4 whitespace-nowrap tabular-nums ${
                                                psi?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''
                                            }`}
                                        >
                                            {psi?.value?.toFixed(4) ?? '—'}
                                        </td>
                                        <td
                                            className={`px-6 py-4 whitespace-nowrap tabular-nums ${
                                                kl?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''
                                            }`}
                                        >
                                            {kl?.value?.toFixed(4) ?? '—'}
                                        </td>
                                        <td
                                            className={`px-6 py-4 whitespace-nowrap tabular-nums ${
                                                ks?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''
                                            }`}
                                        >
                                            {ks?.value?.toFixed(4) ?? '—'}
                                        </td>
                                        {fi && (
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <RiskBadge
                                                    importance={importance}
                                                    rank={importanceRank}
                                                    drifted={col.drift_detected}
                                                    maxImportance={maxImportance}
                                                />
                                            </td>
                                        )}
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <button
                                                onClick={() => toggleRow(col.column)}
                                                className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1 text-sm"
                                            >
                                                {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                                {isExpanded ? 'Hide' : 'Details'}
                                            </button>
                                        </td>
                                        {hasSparklines && (
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <Sparkline values={columnSparklines[col.column] ?? []} />
                                            </td>
                                        )}
                                    </tr>
                                    {isExpanded && (
                                        <tr>
                                            <td colSpan={colSpan} className="px-6 py-4 bg-gray-50 dark:bg-slate-900/50">
                                                <div className="flex flex-col gap-4">
                                                    {col.suggestions && col.suggestions.length > 0 && (
                                                        <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                                                            <div className="flex items-start gap-2">
                                                                <Lightbulb
                                                                    size={16}
                                                                    className="mt-0.5 shrink-0 text-yellow-600 dark:text-yellow-400"
                                                                />
                                                                <ul className="list-disc list-inside text-sm text-yellow-800 dark:text-yellow-200">
                                                                    {col.suggestions.map((s, i) => (
                                                                        <li key={i}>{s}</li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        </div>
                                                    )}
                                                    {col.distribution && <DistributionChart distribution={col.distribution} />}
                                                </div>
                                            </td>
                                        </tr>
                                    )}
                                </React.Fragment>
                            );
                        })
                    )}
                </tbody>
            </table>
        </div>
    );
};
