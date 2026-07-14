import React from 'react';
import type { PreviewDataRows } from '../../../core/api/client';

interface ResultsTableProps {
  columns: string[];
  currentRows: PreviewDataRows;
  effectiveTab: string | null;
}

/** Scrollable preview-rows table for the currently active tab/branch. */
export const ResultsTable: React.FC<ResultsTableProps> = ({ columns, currentRows, effectiveTab }) => (
  <div className="flex-1 overflow-auto">
    <table className="w-full text-sm text-left border-collapse">
      <thead className="text-xs text-muted-foreground uppercase bg-muted sticky top-0 z-10 shadow-sm">
        <tr>
          {columns.map((col) => (
            <th key={col} className="px-4 py-2 font-medium border-b whitespace-nowrap">
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {currentRows.map((row: unknown, idx: number) => (
          <tr key={idx} className="border-b hover:bg-muted/10">
            {columns.map((col) => (
              <td key={`${idx}-${col}`} className="px-4 py-2 whitespace-nowrap font-mono text-xs">
                {(row as Record<string, unknown>)[col] !== null ? String((row as Record<string, unknown>)[col]) : <span className="text-muted-foreground italic">null</span>}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    {currentRows.length === 0 && (
      <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
        No preview data available {effectiveTab ? `for ${effectiveTab}` : ''}
      </div>
    )}
  </div>
);
