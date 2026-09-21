import React from 'react';

function isFiniteValue(value: number | null | undefined): value is number {
    return value != null && Number.isFinite(value);
}

/**
 * Tiny SVG sparkline used in the per-feature drift table to show the recent
 * PSI trend. Gaps retain missing checks; only the endpoint encodes the PSI verdict.
 */
export const Sparkline: React.FC<{
    values: (number | null)[];
    threshold: number | undefined;
    width?: number;
    height?: number;
}> = ({ values, threshold, width = 64, height = 20 }) => {
    if (values.length < 2) return <span className="text-[10px] text-gray-400">—</span>;
    const valid = values.filter(isFiniteValue);
    if (!valid.length) return <span className="text-[10px] text-gray-400" title="PSI unavailable for all checks">—</span>;
    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const range = max - min || 1;
    const segments: string[][] = [];
    let segment: string[] = [];
    const points = values.map((value, index) => {
        if (!isFiniteValue(value)) {
            segment = [];
            return null;
        }
        if (!segment.length) segments.push(segment);
        const point = { x: (index / (values.length - 1)) * width,
            y: height - ((value - min) / range) * (height - 4) - 2 };
        segment.push(`${point.x},${point.y}`);
        return point;
    });
    const last = values[values.length - 1];
    const hasLast = isFiniteValue(last);
    const hasThreshold = isFiniteValue(threshold);
    const color = hasLast && hasThreshold ? (last > threshold ? '#ef4444' : '#22c55e') : '#94a3b8';
    const label = `PSI history: ${values.length - valid.length} missing checks; latest PSI ${hasLast ? last : 'unavailable'}; threshold ${hasThreshold ? threshold : 'unavailable'}`;
    return (
        <svg width={width} height={height} className="inline-block" role="img" aria-label={label}>
            <title>{label}</title>
            {segments.map((part, index) => <polyline key={index} points={part.join(' ')} fill="none" stroke="#3b82f6" strokeWidth="1.5" strokeLinejoin="round" />)}
            {points.map((point, index) => point && <circle key={index} cx={point.x} cy={point.y} r="2" fill={index === values.length - 1 ? color : '#3b82f6'} />)}
        </svg>
    );
};
