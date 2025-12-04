import React, { useState } from 'react';
import { useGraphStore } from '../../core/store/useGraphStore';
import { ChevronUp, ChevronDown, Table, AlertCircle } from 'lucide-react';

export const ResultsPanel: React.FC = () => {
  const executionResult = useGraphStore((state) => state.executionResult);
  const [isExpanded, setIsExpanded] = useState(true);

  if (!executionResult) return null;

  const { preview_data, preview_rows, preview_total_rows, signals } = executionResult;
  const columns = preview_data && preview_data.length > 0 ? Object.keys(preview_data[0]) : [];

  return (
    <div 
      className={`absolute bottom-0 left-64 right-80 bg-background border-t shadow-lg transition-all duration-300 z-20 flex flex-col ${
        isExpanded ? 'h-80' : 'h-10'
      }`}
    >
      {/* Header */}
      <div 
        className="flex items-center justify-between px-4 py-2 bg-muted/10 cursor-pointer hover:bg-muted/20 border-b"
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
          {signals && signals.length > 0 && (
            <div className="p-2 bg-yellow-50 border-b flex gap-2 overflow-x-auto">
              {signals.map((signal: any, idx: number) => (
                <div key={idx} className="flex items-center gap-1 text-xs text-yellow-800 bg-yellow-100 px-2 py-1 rounded border border-yellow-200 whitespace-nowrap">
                  <AlertCircle className="w-3 h-3" />
                  {signal.message || JSON.stringify(signal)}
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
