import React from 'react';
import { Download, Filter, X } from 'lucide-react';
import type { SortConfig } from './_hooks/useSortConfig';

interface DriftFiltersBarProps {
    showOnlyDrifted: boolean;
    onToggleDrifted: () => void;
    sortConfig: SortConfig | null;
    onClearSort: () => void;
    onExport: () => void;
}

/** Toggle for "drifted only", clear-sort link, and CSV export button. */
export const DriftFiltersBar: React.FC<DriftFiltersBarProps> = ({
    showOnlyDrifted,
    onToggleDrifted,
    sortConfig,
    onClearSort,
    onExport,
}) => (
    <div className="flex items-center gap-3 mb-4">
        <button
            onClick={onToggleDrifted}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors border ${
                showOnlyDrifted
                    ? 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700 text-red-700 dark:text-red-300'
                    : 'border-gray-200 dark:border-slate-600 text-gray-500 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-700'
            }`}
        >
            <Filter size={12} />
            {showOnlyDrifted ? 'Showing drifted only' : 'Show only drifted'}
        </button>
        {sortConfig && (
            <button
                onClick={onClearSort}
                className="flex items-center gap-1 px-2 py-1.5 rounded-md text-xs text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 transition-colors"
            >
                <X size={11} /> Clear sort
            </button>
        )}
        <div className="ml-auto">
            <button
                onClick={onExport}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-gray-200 dark:border-slate-600 text-gray-500 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
                title="Download drift report as CSV"
            >
                <Download size={12} /> Export CSV
            </button>
        </div>
    </div>
);
