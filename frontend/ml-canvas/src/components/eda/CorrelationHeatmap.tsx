import React, { useMemo } from 'react';
import type { CorrelationMatrix } from '../../core/types/edaProfile';
import { ChartDataTable } from './ChartDataTable';
import { correlationColor, correlationScopeNotes, CORRELATION_MAX_COLUMNS, CORRELATION_SCALE } from './correlationPresentation';

interface CorrelationHeatmapProps {
  data: CorrelationMatrix;
}

/** Distinguish omitted analysis columns from columns merely hidden by the chart cap. */
const CorrelationOmissions: React.FC<{ data: CorrelationMatrix }> = ({ data }) => <>
  {correlationScopeNotes(data).map(note => (
    <div key={note} className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 text-sm rounded-md">{note}</div>
  ))}
</>;

/** Persistent -1/0/+1 color scale so correlation direction/strength never depends on hover or memory. */
const CorrelationScaleLegend: React.FC = () => (
  <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-gray-600 dark:text-gray-300" role="img" aria-label="Correlation color scale from negative one (blue) through zero (white) to positive one (red)">
    {CORRELATION_SCALE.map(value => <span key={value} className="flex flex-col items-center gap-1">
      <span className="h-3 w-6 rounded-sm border border-gray-300" style={{ backgroundColor: correlationColor(value) }} />
      <span>{value === -1 ? '−1' : value === 1 ? '+1' : value}</span>
    </span>)}
    <span className="ml-2 flex flex-col items-center gap-1">
      <span className="h-3 w-6 rounded-sm border border-gray-300" style={{ backgroundColor: correlationColor(null) }} />
      <span>Missing</span>
    </span>
  </div>
);

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ data }) => {
  // Limit to top 20 columns to prevent crashing on large datasets; the full
  // matrix remains available below via the data-table alternative.
  const displayColumns = useMemo(() => (data?.columns ?? []).slice(0, CORRELATION_MAX_COLUMNS), [data]);
  const displayValues = useMemo(
    () => (data?.values ?? []).slice(0, CORRELATION_MAX_COLUMNS).map((row) => row.slice(0, CORRELATION_MAX_COLUMNS)),
    [data]
  );
  const tableRows = useMemo(() => {
    if (!data?.columns) return [];
    return data.columns.map((rowCol, i) => {
      const row: Record<string, string | number | null> = { row_label: rowCol };
      data.columns.forEach((_, j) => {
        row[`column_${j}`] = data.values[i]?.[j] ?? null;
      });
      return row;
    });
  }, [data]);

  if (!data || !data.columns) return <div>No correlation data available</div>;

  return (
    <div className="overflow-x-auto p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <CorrelationScaleLegend />
      <CorrelationOmissions data={data} />
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
                  style={{ backgroundColor: correlationColor(val), color: '#111827' }}
                  title={`${rowCol} vs ${displayColumns[j]}: ${val !== null ? val.toFixed(3) : 'N/A'}`}
                >
                  {val !== null ? val.toFixed(2) : '—'}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
      <ChartDataTable
        caption="Full correlation matrix data table"
        filename="correlation-matrix"
        columns={[{ key: 'row_label', label: 'Variable' }, ...data.columns.map((col, index) => ({ key: `column_${index}`, label: col }))]}
        rows={tableRows}
      />
    </div>
  );
};
