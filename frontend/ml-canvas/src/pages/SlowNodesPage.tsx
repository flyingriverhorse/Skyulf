/**
 * SlowNodesPage — F4 admin view.
 *
 * Aggregates per-node execution time across completed training/tuning jobs
 * in the lookback window and surfaces the heaviest steps so we know where
 * to invest in optimisation. Reads `metrics.node_timings` (written by
 * `JobStrategy.handle_success`) — no extra instrumentation; jobs that
 * pre-date the field are skipped silently.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    AlertCircle,
    BarChart3,
    Clock,
    Hash,
    Layers,
    RefreshCw,
    Timer,
    TrendingUp,
} from 'lucide-react';
import {
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    ResponsiveContainer,
    Tooltip as RechartsTooltip,
    XAxis,
    YAxis,
} from 'recharts';
import { monitoringApi, SlowNodesResponse } from '../core/api/monitoring';
import { toast } from '../core/toast';

type SortKey = 'total_seconds' | 'avg_seconds' | 'p95_seconds' | 'count' | 'max_seconds';

const WINDOW_OPTIONS: ReadonlyArray<number> = [1, 7, 30, 90];
const LIMIT_OPTIONS: ReadonlyArray<number> = [10, 25, 50];

// Palette cycled across the bar chart so neighbouring step-types are
// visually distinct without overwhelming the eye.
const BAR_COLORS = [
    '#3b82f6',
    '#8b5cf6',
    '#ec4899',
    '#f59e0b',
    '#10b981',
    '#06b6d4',
    '#ef4444',
    '#a855f7',
    '#eab308',
    '#14b8a6',
];

const formatSeconds = (s: number): string => {
    if (s < 1) return `${(s * 1000).toFixed(0)} ms`;
    if (s < 60) return `${s.toFixed(2)} s`;
    const m = Math.floor(s / 60);
    const rem = s - m * 60;
    return `${m}m ${rem.toFixed(1)}s`;
};

export const SlowNodesPage: React.FC = () => {
    const [days, setDays] = useState<number>(7);
    const [limit, setLimit] = useState<number>(10);
    const [data, setData] = useState<SlowNodesResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [sortKey, setSortKey] = useState<SortKey>('total_seconds');

    const load = useCallback(async () => {
        setIsLoading(true);
        setError(null);
        try {
            const resp = await monitoringApi.getSlowNodes(days, limit);
            setData(resp);
        } catch (e) {
            const msg = (e as Error).message || 'Failed to load slow-nodes telemetry';
            setError(msg);
            toast.error(msg);
        } finally {
            setIsLoading(false);
        }
    }, [days, limit]);

    useEffect(() => {
        void load();
    }, [load]);

    /** Sort copy of aggregates by the user-selected column. */
    const sortedAggregates = useMemo(() => {
        if (!data) return [];
        return [...data.aggregates].sort((a, b) => b[sortKey] - a[sortKey]);
    }, [data, sortKey]);

    /** Peak total for proportional bar widths in the table. */
    const peakTotal = useMemo(() => {
        if (sortedAggregates.length === 0) return 1;
        return Math.max(...sortedAggregates.map(a => a.total_seconds));
    }, [sortedAggregates]);

    return (
        <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 p-6 overflow-auto">
            <div className="flex items-start justify-between flex-wrap gap-4 mb-6">
                <div>
                    <h1 className="text-2xl font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
                        <TrendingUp className="w-6 h-6 text-blue-500" />
                        Slow Nodes
                    </h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        Workspace-wide step-type runtime aggregated from completed jobs in the last
                        window. Use it to spot the cheapest optimisation wins.
                    </p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                    <div className="inline-flex rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                        {WINDOW_OPTIONS.map(d => (
                            <button
                                key={d}
                                onClick={() => setDays(d)}
                                className={`px-2.5 py-1 text-xs font-medium tabular-nums border-r last:border-r-0 border-gray-200 dark:border-gray-700 transition-colors ${
                                    d === days
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/30'
                                }`}
                            >
                                {d === 1 ? '24 h' : `${d} d`}
                            </button>
                        ))}
                    </div>
                    <div className="inline-flex rounded-md overflow-hidden border border-gray-200 dark:border-gray-700">
                        {LIMIT_OPTIONS.map(n => (
                            <button
                                key={n}
                                onClick={() => setLimit(n)}
                                className={`px-2.5 py-1 text-xs font-medium tabular-nums border-r last:border-r-0 border-gray-200 dark:border-gray-700 transition-colors ${
                                    n === limit
                                        ? 'bg-blue-600 text-white'
                                        : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/30'
                                }`}
                                title={`Show top ${n} step types`}
                            >
                                Top {n}
                            </button>
                        ))}
                    </div>
                    <button
                        onClick={() => void load()}
                        disabled={isLoading}
                        className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
                        Refresh
                    </button>
                </div>
            </div>

            {/* Header stats */}
            {data && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                    <StatCard
                        icon={<Layers className="w-4 h-4 text-blue-500" />}
                        label="Step types"
                        value={String(data.aggregates.length)}
                    />
                    <StatCard
                        icon={<Hash className="w-4 h-4 text-blue-500" />}
                        label="Node runs"
                        value={data.total_node_runs.toLocaleString()}
                    />
                    <StatCard
                        icon={<BarChart3 className="w-4 h-4 text-blue-500" />}
                        label="Jobs scanned"
                        value={data.total_jobs_scanned.toLocaleString()}
                    />
                    <StatCard
                        icon={<Clock className="w-4 h-4 text-blue-500" />}
                        label="Window"
                        value={`${data.days} day${data.days === 1 ? '' : 's'}`}
                    />
                </div>
            )}

            {error && (
                <div className="mb-4 flex items-start gap-2 p-3 rounded border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                </div>
            )}

            {sortedAggregates.length > 0 && (
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-6">
                    <div className="flex items-center gap-2 mb-3">
                        <BarChart3 className="w-4 h-4 text-blue-500" />
                        <h2 className="text-sm font-semibold text-gray-700 dark:text-gray-200">
                            Total time by step type
                        </h2>
                        <span className="text-[11px] text-gray-400">
                            ({sortKey === 'total_seconds' ? 'sorted by total' : `sorted by ${sortKey.replace('_seconds', '')}`})
                        </span>
                    </div>
                    <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={sortedAggregates.map(a => ({
                                    step_type: a.step_type,
                                    total: Number(a.total_seconds.toFixed(3)),
                                    avg: Number(a.avg_seconds.toFixed(3)),
                                    runs: a.count,
                                }))}
                                margin={{ top: 4, right: 8, left: 8, bottom: 28 }}
                            >
                                <CartesianGrid
                                    strokeDasharray="3 3"
                                    className="stroke-gray-200 dark:stroke-slate-700"
                                    vertical={false}
                                />
                                <XAxis
                                    dataKey="step_type"
                                    tick={{ fill: '#64748b', fontSize: 10 }}
                                    angle={-25}
                                    textAnchor="end"
                                    interval={0}
                                    height={50}
                                />
                                <YAxis
                                    tick={{ fill: '#64748b', fontSize: 10 }}
                                    tickFormatter={(v: number) => formatSeconds(v)}
                                    width={64}
                                />
                                <RechartsTooltip
                                    contentStyle={{
                                        backgroundColor: '#1e293b',
                                        borderColor: '#334155',
                                        color: '#f8fafc',
                                        borderRadius: 6,
                                        fontSize: 12,
                                    }}
                                    itemStyle={{ color: '#f8fafc' }}
                                    labelStyle={{ color: '#94a3b8' }}
                                    formatter={(value: number, name: string) => {
                                        if (name === 'runs') return [value, 'Runs'];
                                        return [formatSeconds(value), name === 'total' ? 'Total time' : 'Avg per run'];
                                    }}
                                />
                                <Bar dataKey="total" radius={[3, 3, 0, 0]}>
                                    {sortedAggregates.map((_, idx) => (
                                        <Cell
                                            key={`bar-${idx}`}
                                            fill={BAR_COLORS[idx % BAR_COLORS.length]}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="bg-gray-50 dark:bg-gray-900/40 border-b border-gray-200 dark:border-gray-700">
                            <tr>
                                <th
                                    scope="col"
                                    className="text-left px-3 py-2 text-[11px] uppercase tracking-wider font-semibold text-gray-500 dark:text-gray-400"
                                >
                                    Step type
                                </th>
                                <SortHeader
                                    k="count"
                                    label="Runs"
                                    title="Number of executions"
                                    sortKey={sortKey}
                                    onSort={setSortKey}
                                />
                                <SortHeader
                                    k="total_seconds"
                                    label="Total"
                                    title="Cumulative time across all runs"
                                    sortKey={sortKey}
                                    onSort={setSortKey}
                                />
                                <SortHeader
                                    k="avg_seconds"
                                    label="Avg"
                                    title="Mean wall-clock time per run"
                                    sortKey={sortKey}
                                    onSort={setSortKey}
                                />
                                <SortHeader
                                    k="p95_seconds"
                                    label="p95"
                                    title="95th-percentile worst-case run"
                                    sortKey={sortKey}
                                    onSort={setSortKey}
                                />
                                <SortHeader
                                    k="max_seconds"
                                    label="Max"
                                    title="Slowest single run"
                                    sortKey={sortKey}
                                    onSort={setSortKey}
                                />
                                <th
                                    scope="col"
                                    className="text-left px-3 py-2 text-[11px] uppercase tracking-wider font-semibold text-gray-500 dark:text-gray-400"
                                    aria-label="Distribution"
                                >
                                    Share
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {isLoading && !data ? (
                                <tr>
                                    <td
                                        colSpan={7}
                                        className="px-3 py-8 text-center text-gray-400 italic"
                                    >
                                        Loading…
                                    </td>
                                </tr>
                            ) : sortedAggregates.length === 0 ? (
                                <tr>
                                    <td
                                        colSpan={7}
                                        className="px-3 py-8 text-center text-gray-400"
                                    >
                                        <div className="flex flex-col items-center gap-2">
                                            <Timer className="w-8 h-8 opacity-40" />
                                            <span className="italic">
                                                No node-timing data in the last {days} day
                                                {days === 1 ? '' : 's'}.
                                            </span>
                                            <span className="text-[11px] not-italic text-gray-500">
                                                Pre-existing jobs are silently skipped — re-run a
                                                pipeline to seed this view.
                                            </span>
                                        </div>
                                    </td>
                                </tr>
                            ) : (
                                sortedAggregates.map(agg => {
                                    const sharePct = (agg.total_seconds / peakTotal) * 100;
                                    return (
                                        <tr
                                            key={agg.step_type}
                                            className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50"
                                        >
                                            <td className="px-3 py-2">
                                                <div className="font-mono font-medium text-gray-800 dark:text-gray-100">
                                                    {agg.step_type}
                                                </div>
                                                {agg.sample_node_id && (
                                                    <div
                                                        className="text-[10px] text-gray-400 truncate max-w-[260px]"
                                                        title={agg.sample_node_id}
                                                    >
                                                        e.g. {agg.sample_node_id}
                                                    </div>
                                                )}
                                            </td>
                                            <td className="px-3 py-2 text-right tabular-nums text-gray-700 dark:text-gray-200">
                                                {agg.count.toLocaleString()}
                                            </td>
                                            <td className="px-3 py-2 text-right tabular-nums font-medium text-gray-800 dark:text-gray-100">
                                                {formatSeconds(agg.total_seconds)}
                                            </td>
                                            <td className="px-3 py-2 text-right tabular-nums text-gray-700 dark:text-gray-200">
                                                {formatSeconds(agg.avg_seconds)}
                                            </td>
                                            <td className="px-3 py-2 text-right tabular-nums text-gray-700 dark:text-gray-200">
                                                {formatSeconds(agg.p95_seconds)}
                                            </td>
                                            <td className="px-3 py-2 text-right tabular-nums text-gray-700 dark:text-gray-200">
                                                {formatSeconds(agg.max_seconds)}
                                            </td>
                                            <td className="px-3 py-2 w-48">
                                                <div className="flex items-center gap-2">
                                                    <div className="flex-1 h-2 bg-gray-100 dark:bg-gray-700/50 rounded overflow-hidden">
                                                        <div
                                                            className="h-full bg-blue-500"
                                                            style={{ width: `${sharePct}%` }}
                                                        />
                                                    </div>
                                                    <span className="text-[10px] text-gray-400 tabular-nums w-10 text-right">
                                                        {sharePct.toFixed(0)}%
                                                    </span>
                                                </div>
                                            </td>
                                        </tr>
                                    );
                                })
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

const StatCard: React.FC<{ icon: React.ReactNode; label: string; value: string }> = ({
    icon,
    label,
    value,
}) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3 flex items-center gap-3">
        <div className="shrink-0 w-8 h-8 rounded bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center">
            {icon}
        </div>
        <div className="min-w-0">
            <div className="text-[10px] uppercase tracking-wider text-gray-500 dark:text-gray-400">
                {label}
            </div>
            <div className="font-semibold text-gray-800 dark:text-gray-100 tabular-nums truncate">
                {value}
            </div>
        </div>
    </div>
);

/** Sortable column header — hoisted so React doesn't remount on every render. */
const SortHeader: React.FC<{
    k: SortKey;
    label: string;
    title?: string;
    sortKey: SortKey;
    onSort: (k: SortKey) => void;
}> = ({ k, label, title, sortKey, onSort }) => (
    <th
        scope="col"
        className="text-right px-3 py-2 text-[11px] uppercase tracking-wider font-semibold text-gray-500 dark:text-gray-400 cursor-pointer select-none"
        onClick={() => onSort(k)}
        title={title}
    >
        <span
            className={`inline-flex items-center gap-1 ${
                sortKey === k ? 'text-blue-600 dark:text-blue-300' : ''
            }`}
        >
            {label}
            {sortKey === k && <span aria-hidden="true">▾</span>}
        </span>
    </th>
);
