import React from 'react';

interface CorrelationHeatmapProps {
  data: {
    columns: string[];
    values: number[][];
  };
}

export const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({ data }) => {
  if (!data || !data.columns) return <div>No correlation data available</div>;

  // Limit to top 20 columns to prevent crashing on large datasets
  const MAX_COLS = 20;
  const displayColumns = data.columns.slice(0, MAX_COLS);
  const displayValues = data.values.slice(0, MAX_COLS).map(row => row.slice(0, MAX_COLS));
  
  const isTruncated = data.columns.length > MAX_COLS;

  const getColor = (value: number) => {
    // -1 (blue) -> 0 (white) -> 1 (red)
    if (value === null) return 'rgba(243, 244, 246, 1)'; // gray-100
    
    // Use opacity for intensity
    const opacity = Math.abs(value);
    
    if (value > 0) {
      // Red
      return `rgba(239, 68, 68, ${opacity})`; 
    } else {
      // Blue
      return `rgba(59, 130, 246, ${opacity})`;
    }
  };

  return (
    <div className="overflow-x-auto p-4 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      {isTruncated && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-200 text-sm rounded-md">
          Showing top {MAX_COLS} columns. Full heatmap is too large to render.
        </div>
      )}
      <div className="inline-block min-w-full">
        <div 
            className="grid gap-1" 
            style={{ 
                gridTemplateColumns: `100px repeat(${displayColumns.length}, minmax(50px, 1fr))` 
            }}
        >
          {/* Header Row */}
          <div className="p-2"></div>
          {displayColumns.map((col, i) => (
            <div key={i} className="p-2 text-xs font-medium truncate text-center" title={col}>
              {col.length > 8 ? col.substring(0, 8) + '...' : col}
            </div>
          ))}

          {/* Rows */}
          {displayColumns.map((rowCol, i) => (
            <React.Fragment key={i}>
              {/* Row Label */}
              <div className="p-2 text-xs font-medium truncate text-right pr-4" title={rowCol}>
                {rowCol}
              </div>
              {/* Cells */}
              {displayValues[i].map((val, j) => (
                <div 
                  key={j} 
                  className="h-10 w-full flex items-center justify-center text-[10px] text-gray-900 dark:text-gray-100 rounded-sm cursor-help"
                  style={{ backgroundColor: getColor(val) }}
                  title={`${rowCol} vs ${displayColumns[j]}: ${val.toFixed(3)}`}
                >
                  {Math.abs(val) > 0.3 ? val.toFixed(1) : ''}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};
