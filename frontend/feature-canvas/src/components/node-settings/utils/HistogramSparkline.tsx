import React, { useId } from 'react';

export type HistogramSparklineProps = {
  counts: number[];
  binEdges: number[];
  className?: string;
};

export const HistogramSparkline: React.FC<HistogramSparklineProps> = ({ counts, binEdges, className }) => {
  const gradientId = useId();
  const wrapperClass = className ?? 'canvas-skewness__histogram';
  if (!counts || counts.length === 0 || !binEdges || binEdges.length !== counts.length + 1) {
    return <div className={wrapperClass} aria-hidden="true" />;
  }

  const maxCount = counts.reduce((acc, value) => Math.max(acc, value), 0);
  const safeMax = maxCount > 0 ? maxCount : 1;
  const binWidth = 100 / counts.length;

  return (
    <div className={wrapperClass} role="img" aria-label="Histogram sparkline">
      <svg viewBox="0 0 100 60">
        <line x1="0" y1="58" x2="100" y2="58" stroke="rgba(148, 163, 184, 0.35)" strokeWidth="0.75" />
        {counts.map((count, index) => {
          const height = (count / safeMax) * 52;
          const x = index * binWidth;
          const y = 58 - height;
          const width = Math.max(binWidth - 2, 1);
          return (
            <rect
              key={`hist-bin-${index}`}
              x={x + 1}
              y={y}
              width={width}
              height={height}
              rx={1.2}
              ry={1.2}
              fill={`url(#${gradientId})`}
            />
          );
        })}
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(165, 180, 252, 0.95)" />
            <stop offset="100%" stopColor="rgba(129, 140, 248, 0.35)" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
};
