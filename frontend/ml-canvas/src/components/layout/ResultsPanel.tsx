import React, { useState } from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { ChevronUp, ChevronDown, Table } from 'lucide-react';

export const ResultsPanel: React.FC = () => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const [isExpanded, setIsExpanded] = useState(true);

  if (!executionResult) return null;

  // Map backend response to UI expectations
  // Backend returns: { sample_rows: [], metrics: { preview_rows: 100, preview_total_rows: 1000 }, signals: ... }
  const preview_data = executionResult.sample_rows || [];
  const preview_rows = executionResult.metrics?.preview_rows || 0;
  const preview_total_rows = executionResult.metrics?.preview_total_rows || 0;
  // const signals = executionResult.signals ? [executionResult.signals] : []; // Signals might be an object, wrap in array if needed or iterate keys

  const columns = preview_data && preview_data.length > 0 ? Object.keys(preview_data[0]) : [];

  return (
    <div 
      className={`absolute bottom-0 left-0 right-0 bg-background border-t shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.1)] transition-all duration-300 z-20 flex flex-col ${
        isExpanded ? 'h-96' : 'h-10'
      }`}
    >
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 bg-muted/10 cursor-pointer hover:bg-muted/20 border-b select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Table className="w-4 h-4 text-primary" />
          <span className="font-semibold text-sm">Preview Results</span>
          <span className="text-xs text-muted-foreground ml-2">
            {preview_rows} rows shown (of {preview_total_rows})
          </span>
        </div>
        <button className="p-1 hover:bg-muted rounded">
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </button>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="flex-1 overflow-hidden flex flex-col">
          {/* Signals / Warnings */}
          {/* TODO: Better signal visualization based on signal structure */}
          {executionResult.applied_steps && executionResult.applied_steps.length > 0 && (
             <div className="p-2 bg-blue-50 border-b flex gap-2 overflow-x-auto">
                {executionResult.applied_steps.map((step: string, idx: number) => (
                  <div key={idx} className="text-xs text-blue-800 bg-blue-100 px-2 py-1 rounded border border-blue-200 whitespace-nowrap">
                    {step}
                  </div>
                ))}
             </div>
          )}

          {/* Data Table */}
          <div className="flex-1 overflow-auto">
            <table className="w-full text-sm text-left border-collapse">
              <thead className="bg-muted/50 sticky top-0 z-10">
                <tr>
                  {columns.map((col) => (
                    <th key={col} className="px-4 py-2 font-medium text-muted-foreground border-b border-r last:border-r-0 whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview_data.map((row: any, idx: number) => (
                  <tr key={idx} className="hover:bg-muted/5 border-b last:border-b-0">
                    {columns.map((col) => (
                      <td key={`${idx}-${col}`} className="px-4 py-2 border-r last:border-r-0 whitespace-nowrap font-mono text-xs">
                        {row[col] !== null ? String(row[col]) : <span className="text-muted-foreground italic">null</span>}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
