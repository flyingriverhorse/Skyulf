import React from 'react';

/**
 * Tiny SVG sparkline used in the per-feature drift table to show the recent
 * PSI trend. Endpoint dot is colour-coded (green ≤ 0.1, amber ≤ 0.2, red >).
 */
export const Sparkline: React.FC<{
    values: number[];
    width?: number;
    height?: number;
}> = ({ values, width = 64, height = 20 }) => {
    if (values.length < 2) return <span className="text-[10px] text-gray-400">—</span>;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const points = values
        .map((v, i) => {
            const x = (i / (values.length - 1)) * width;
            const y = height - ((v - min) / range) * (height - 4) - 2;
            return `${x},${y}`;
        })
        .join(' ');
    const last = values[values.length - 1] ?? 0;
    const color = last > 0.2 ? '#ef4444' : last > 0.1 ? '#f59e0b' : '#22c55e';
    const lastY = height - ((last - min) / range) * (height - 4) - 2;
    return (
        <svg width={width} height={height} className="inline-block">
            <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
            <circle cx={width} cy={lastY} r="2" fill={color} />
        </svg>
    );
};
