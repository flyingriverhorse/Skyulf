import React, { useMemo } from 'react';
import { ChartDataTable } from './ChartDataTable';

interface CorrelationHeatmapProps {
  data: {
    columns: string[];
    values: number[][];
  };
}

/** -1 (blue) -> 0 (white) -> 1 (red) cell fill, shared by the grid and the legend swatches. */
const getColor = (value: number | null): string => {
  if (value === null || Number.isNaN(value)) return 'rgba(243, 244, 246, 1)'; // gray-100
  // Use opacity for intensity, but ensure minimum visibility: 0.1 -> 0.4, 1.0 -> 1.0
  const opacity = Math.max(0.4, Math.abs(value));
  return value > 0 ? `rgba(239, 68, 68, ${opacity})` : `rgba(59, 130, 246, ${opacity})`;
};

/** Persistent -1/0/+1 color scale so correlation direction/strength never depends on hover or memory. */
const CorrelationScaleLegend: React.FC = () => (
  <div className="mb-4 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300" role="img" aria-label="Correlation color scale from negative one (blue) through zero (white) to positive one (red)">
    <span>−1</span>
    <div
      className="h-3 w-40 rounded-sm border border-gray-200 dark:border-gray-700"
      style={{ background: 'linear-gradient(to right, rgba(59,130,246,1), rgba(59,130,246,0.4), rgba(243,244,246,1), rgba(239,68,68,0.4), rgba(239,68,68,1))' }}
      aria-hidden="true"
    />
    <span>0</span>
    <span className="w-6" />
    <span>+1</span>
    <span className="ml-2 italic text-gray-400 dark:text-gray-500">Negative (blue) · Positive (red)</span>
  </div>
);

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ data }) => {
  // Limit to top 20 columns to prevent crashing on large datasets; the full
  // matrix remains available below via the data-table alternative.
  const MAX_COLS = 20;

  const displayColumns = useMemo(() => (data?.columns ?? []).slice(0, MAX_COLS), [data]);
  const displayValues = useMemo(
    () => (data?.values ?? []).slice(0, MAX_COLS).map((row) => row.slice(0, MAX_COLS)),
    [data]
  );
  const omittedColumns = useMemo(() => (data?.columns ?? []).slice(MAX_COLS), [data]);

  const tableRows = useMemo(() => {
    if (!data?.columns) return [];
    return data.columns.map((rowCol, i) => {
      const row: Record<string, string | number | null> = { variable: rowCol };
      data.columns.forEach((colName, j) => {
        row[colName] = data.values[i]?.[j] ?? null;
      });
      return row;
    });
  }, [data]);

  if (!data || !data.columns) return <div>No correlation data available</div>;

  const isTruncated = data.columns.length > MAX_COLS;

  return (
    <div className="overflow-x-auto p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <CorrelationScaleLegend />
      {isTruncated && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 text-sm rounded-md">
          Showing the first {MAX_COLS} of {data.columns.length} columns
          ({omittedColumns.length} omitted: {omittedColumns.slice(0, 5).join(', ')}
          {omittedColumns.length > 5 ? ', …' : ''}). Use the data table below for the full matrix.
        </div>
      )}
      <div className="inline-block min-w-full">
        <div
            className="grid gap-1"
            style={{
                gridTemplateColumns: `100px repeat(${displayColumns.length}, minmax(50px, 1fr))`,
                paddingTop: '4.5rem'
            }}
        >
          {/* Header Row — labels are rotated so full names are visible without hovering. */}
          <div className="p-2"></div>
          {displayColumns.map((col, i) => (
            <div
              key={i}
              className="relative p-2 text-xs font-medium text-gray-700 dark:text-gray-300"
              title={col}
            >
              <span
                className="absolute bottom-1 left-1/2 whitespace-nowrap origin-bottom-left"
                style={{ transform: 'rotate(-45deg)' }}
              >
                {col}
              </span>
            </div>
          ))}

          {/* Rows */}
          {displayColumns.map((rowCol, i) => (
            <React.Fragment key={i}>
              {/* Row Label */}
              <div className="p-2 text-xs font-medium truncate text-right pr-4 text-gray-700 dark:text-gray-300" title={rowCol}>
                {rowCol}
              </div>
              {/* Cells */}
              {(displayValues[i] ?? []).map((val, j) => (
                <div
                  key={j}
                  className="h-10 w-full flex items-center justify-center text-[10px] text-gray-900 dark:text-gray-100 rounded-sm cursor-help transition-opacity hover:opacity-80"
                  style={{ backgroundColor: getColor(val) }}
                  title={`${rowCol} vs ${displayColumns[j]}: ${val !== null ? val.toFixed(3) : 'N/A'}`}
                >
                  {val !== null ? val.toFixed(2) : ''}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
      <ChartDataTable
        caption="Full correlation matrix data table"
        filename="correlation-matrix"
        columns={[{ key: 'variable', label: 'Variable' }, ...data.columns.map((col) => ({ key: col, label: col }))]}
        rows={tableRows}
      />
    </div>
  );
};
