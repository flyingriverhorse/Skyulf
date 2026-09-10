import React, { useMemo } from 'react';
import { HISTOGRAM_BINS, renderPrediction, SAMPLE_OPTIONS } from './inferenceData';


/** Tiny inline SVG histogram of numeric predictions — no recharts dep. */
export const PredictionHistogram: React.FC<{ values: number[] }> = ({ values }) => {
    const stats = useMemo(() => {
        if (values.length === 0) return null;
        const min = Math.min(...values);
        const max = Math.max(...values);
        if (min === max) return { min, max, bins: [values.length] as number[] };
        const binWidth = (max - min) / HISTOGRAM_BINS;
        const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
        values.forEach(v => {
            let idx = Math.floor((v - min) / binWidth);
            if (idx >= HISTOGRAM_BINS) idx = HISTOGRAM_BINS - 1;
            const safe = bins[idx];
            bins[idx] = (safe ?? 0) + 1;
        });
        return { min, max, bins };
    }, [values]);

    if (!stats) return null;
    const width = 220;
    const height = 48;
    const peak = Math.max(...stats.bins, 1);
    const barWidth = width / stats.bins.length;

    return (
        <div className="flex items-center gap-2" title="Distribution of predictions">
            <svg width={width} height={height} className="text-blue-500">
                {stats.bins.map((count, i) => {
                    const h = (count / peak) * (height - 4);
                    return (
                        <rect
                            key={i}
                            x={i * barWidth + 1}
                            y={height - h}
                            width={Math.max(1, barWidth - 2)}
                            height={h}
                            fill="currentColor"
                            opacity={0.85}
                        />
                    );
                })}
            </svg>
            <div className="flex flex-col text-[10px] text-gray-400 leading-tight">
                <span className="tabular-nums">{stats.max.toFixed(2)}</span>
                <span className="tabular-nums">{stats.min.toFixed(2)}</span>
            </div>
        </div>
    );
};

/** Sparkline of recent-run latencies — surfaced in the "Recent runs" header. */
export const LatencySparkline: React.FC<{ values: number[] }> = ({ values }) => {
    if (values.length < 2) return null;
    const width = 60;
    const height = 16;
    const max = Math.max(...values);
    const min = Math.min(...values);
    const range = max - min || 1;
    const points = values
        .map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 2) - 1;
            return `${x},${y}`;
        })
        .join(' ');
    return (
        <svg width={width} height={height} className="text-blue-400" aria-hidden="true">
            <polyline
                points={points}
                fill="none"
                stroke="currentColor"
                strokeWidth="1.25"
                strokeLinejoin="round"
            />
        </svg>
    );
};

/** Bar visualisation for a {class: probability} prediction. */
export const ProbabilityBars: React.FC<{ probs: Record<string, number> }> = ({ probs }) => {
    const entries = Object.entries(probs).sort(([, a], [, b]) => b - a);
    const peak = Math.max(...entries.map(([, v]) => v), 1);
    return (
        <div className="flex flex-col gap-1 w-full">
            {entries.map(([cls, p], i) => {
                const pct = (p / peak) * 100;
                const isTop = i === 0;
                return (
                    <div key={cls} className="flex items-center gap-2 text-[11px]">
                        <span
                            className={`font-mono shrink-0 w-20 truncate ${isTop
                                    ? 'text-blue-700 dark:text-blue-300 font-semibold'
                                    : 'text-gray-500 dark:text-gray-400'
                                }`}
                            title={cls}
                        >
                            {cls}
                        </span>
                        <div className="flex-1 h-2 bg-gray-100 dark:bg-gray-700/50 rounded overflow-hidden">
                            <div
                                className={`h-full ${isTop ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
                                    }`}
                                style={{ width: `${pct}%` }}
                            />
                        </div>
                        <span className="tabular-nums w-12 text-right text-gray-600 dark:text-gray-300">
                            {(p * 100).toFixed(1)}%
                        </span>
                    </div>
                );
            })}
        </div>
    );
};

/** Side-by-side table view: each input row beside its prediction. */
export const InputOutputTable: React.FC<{ rows: unknown[]; predictions: unknown[] }> = ({
    rows,
    predictions,
}) => {
    const keys = useMemo(() => {
        const set = new Set<string>();
        rows.forEach(r => {
            if (r && typeof r === 'object') {
                Object.keys(r as Record<string, unknown>).forEach(k => set.add(k));
            }
        });
        return [...set];
    }, [rows]);

    if (rows.length === 0) return null;

    return (
        <div className="overflow-auto h-full">
            <table className="w-full text-xs border-collapse">
                <thead className="sticky top-0 bg-gray-50 dark:bg-gray-900">
                    <tr>
                        <th className="text-left px-2 py-1 text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700 w-8">
                            #
                        </th>
                        {keys.map(k => (
                            <th
                                key={k}
                                className="text-left px-2 py-1 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-200 dark:border-gray-700"
                            >
                                {k}
                            </th>
                        ))}
                        <th className="text-left px-2 py-1 text-blue-500 font-semibold border-b border-gray-200 dark:border-gray-700 sticky right-0 bg-gray-50 dark:bg-gray-900">
                            prediction
                        </th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => {
                        const obj = (row && typeof row === 'object'
                            ? (row as Record<string, unknown>)
                            : {}) as Record<string, unknown>;
                        return (
                            <tr
                                key={i}
                                className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/40"
                            >
                                <td className="px-2 py-1 text-gray-400 tabular-nums">{i + 1}</td>
                                {keys.map(k => (
                                    <td
                                        key={k}
                                        className="px-2 py-1 font-mono text-gray-700 dark:text-gray-300 truncate max-w-[140px]"
                                        title={String(obj[k] ?? '')}
                                    >
                                        {obj[k] === undefined ? (
                                            <span className="text-gray-300 dark:text-gray-600">—</span>
                                        ) : (
                                            String(obj[k])
                                        )}
                                    </td>
                                ))}
                                <td className="px-2 py-1 font-mono font-medium text-blue-600 dark:text-blue-400 sticky right-0 bg-white dark:bg-gray-800">
                                    {renderPrediction(predictions[i])}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};

/** Polished segmented control for picking the sample-row count. */
export const SampleSizeSegmented: React.FC<{
    value: number;
    onChange: (n: number) => void;
    disabled: boolean;
}> = ({ value, onChange, disabled }) => (
    <div
        className={`inline-flex rounded-md overflow-hidden border ${disabled
                ? 'border-gray-200 dark:border-gray-700 opacity-50 cursor-not-allowed'
                : 'border-gray-200 dark:border-gray-600'
            }`}
        role="group"
        aria-label="Sample row count"
    >
        {SAMPLE_OPTIONS.map(n => {
            const active = n === value;
            return (
                <button
                    key={n}
                    type="button"
                    disabled={disabled}
                    onClick={() => onChange(n)}
                    className={`px-2 py-1 text-[11px] font-medium tabular-nums transition-colors border-r last:border-r-0 ${disabled ? 'border-gray-200 dark:border-gray-700' : 'border-gray-200 dark:border-gray-600'
                        } ${active
                            ? 'bg-blue-600 text-white'
                            : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/30'
                        }`}
                    title={`Fetch ${n} sample row${n === 1 ? '' : 's'}`}
                >
                    {n}
                </button>
            );
        })}
    </div>
);
