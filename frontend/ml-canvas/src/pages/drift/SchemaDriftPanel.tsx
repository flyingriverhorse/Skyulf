import React from 'react';
import { AlertTriangle, ChevronDown, Info } from 'lucide-react';

interface SchemaDriftPanelProps {
    missingColumns: string[];
    newColumns: string[];
}

/**
 * Two collapsible cards showing columns that disappeared from the upload
 * (vs. the training reference) and columns that newly appeared. Renders
 * nothing when both lists are empty.
 */
export const SchemaDriftPanel: React.FC<SchemaDriftPanelProps> = ({ missingColumns, newColumns }) => {
    const missing = missingColumns.filter(c => c.trim() !== '');
    const added = newColumns.filter(c => c.trim() !== '');
    if (missing.length === 0 && added.length === 0) return null;

    return (
        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            {missing.length > 0 && (
                <details className="group bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
                    <summary className="font-semibold text-red-800 dark:text-red-200 p-4 flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden">
                        <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
                        <AlertTriangle size={16} className="shrink-0" /> Missing Columns
                        <span className="ml-auto text-xs font-normal bg-red-200 dark:bg-red-800 px-2 py-0.5 rounded-full">
                            {missing.length}
                        </span>
                    </summary>
                    <div className="px-4 pb-4">
                        <p className="text-[11px] text-red-600/70 dark:text-red-300/60 mb-2">
                            Columns present in reference (training) data but missing from your uploaded file.
                        </p>
                        <ul className="list-disc list-inside text-sm text-red-700 dark:text-red-300">
                            {missing.map(col => (
                                <li key={col}>{col}</li>
                            ))}
                        </ul>
                    </div>
                </details>
            )}
            {added.length > 0 && (
                <details className="group bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                    <summary className="font-semibold text-blue-800 dark:text-blue-200 p-4 flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden">
                        <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
                        <Info size={16} className="shrink-0" /> Extra Columns
                        <span className="ml-auto text-xs font-normal bg-blue-200 dark:bg-blue-800 px-2 py-0.5 rounded-full">
                            {added.length}
                        </span>
                    </summary>
                    <div className="px-4 pb-4">
                        <p className="text-[11px] text-blue-600/70 dark:text-blue-300/60 mb-2">
                            Columns in your uploaded file that were not in the training data (e.g. target column,
                            IDs, or dropped features). These are ignored during drift analysis.
                        </p>
                        <ul className="list-disc list-inside text-sm text-blue-700 dark:text-blue-300">
                            {added.map(col => (
                                <li key={col}>{col}</li>
                            ))}
                        </ul>
                    </div>
                </details>
            )}
        </div>
    );
};
