import React from 'react';
import { AuditRow } from './AuditRow';
import type { AuditLogState } from './useAuditLog';

/** Preserve loading, dataset, empty, and newest-first row presentation. */
export const AuditHistory: React.FC<{ state: AuditLogState }> = ({ state }) => {
    const { isLoading, datasetId, data, emptyStateText, filteredEntries } = state;
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            {isLoading ? (
                <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    Loading audit trail…
                </div>
            ) : !datasetId ? (
                <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    Pick a dataset to view its save history.
                </div>
            ) : !data || data.entries.length === 0 ? (
                <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    {emptyStateText}
                </div>
            ) : filteredEntries.length === 0 ? (
                <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    {emptyStateText}
                </div>
            ) : (
                <div>
                    {filteredEntries.map((entry, idx) => (
                        <AuditRow
                            key={entry.id}
                            entry={entry}
                            // entries is newest-first; the chronologically-first save is
                            // the LAST element in the array.
                            isFirst={idx === filteredEntries.length - 1}
                        />
                    ))}
                </div>
            )}
        </div>
    );
};
