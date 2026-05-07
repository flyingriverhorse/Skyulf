import React from 'react';
import { Activity } from 'lucide-react';

/** Centred placeholder shown when no drift report has been generated yet. */
export const EmptyState: React.FC = () => (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow border dark:border-slate-700 p-12 text-center">
        <div className="flex flex-col items-center gap-3 text-gray-400 dark:text-slate-500">
            <Activity size={48} strokeWidth={1.2} />
            <h2 className="text-lg font-medium text-gray-500 dark:text-slate-400">No Drift Report Yet</h2>
            <p className="text-sm max-w-md">
                Select a reference model from the dropdown above, upload current production data (CSV or
                Parquet), and click <strong>Run Analysis</strong> to compare distributions.
            </p>
        </div>
    </div>
);
