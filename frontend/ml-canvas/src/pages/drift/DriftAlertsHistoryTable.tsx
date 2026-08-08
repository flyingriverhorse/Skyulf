import React from 'react';
import { Search, TrendingUp } from 'lucide-react';
import type { DriftHistoryEntry } from '../../core/api/monitoring';
import { DriftSeverityBadge, DriftStatusBadge } from './DriftAlertBadges';

interface DriftAlertsHistoryTableProps {
    history: DriftHistoryEntry[];
    onInvestigate: (alertId: number) => void;
}

/**
 * Lists every persisted drift check for the selected job — including
 * `no_baseline`/`failed` evaluations that never produced a report — with its
 * severity, disposition, and threshold version, and an Investigate action
 * that opens the full alert detail (OPS-003 durable alert history).
 */
export const DriftAlertsHistoryTable: React.FC<DriftAlertsHistoryTableProps> = ({
    history,
    onInvestigate,
}) => {
    if (history.length === 0) return null;

    return (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow border dark:border-slate-700 mt-6">
            <div className="p-4 border-b border-gray-200 dark:border-slate-700">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                    <TrendingUp size={18} /> Alert History
                    <span className="text-xs font-normal text-gray-400 ml-1">
                        ({history.length} checks)
                    </span>
                </h2>
            </div>
            <div className="overflow-x-auto">
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-left text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-slate-700">
                            <th className="px-4 py-2 font-medium">Detected</th>
                            <th className="px-4 py-2 font-medium">Severity</th>
                            <th className="px-4 py-2 font-medium">Status</th>
                            <th className="px-4 py-2 font-medium">Evaluation</th>
                            <th className="px-4 py-2 font-medium">Threshold</th>
                            <th className="px-4 py-2 font-medium">Owner</th>
                            <th className="px-4 py-2 font-medium sr-only">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {history.map(entry => (
                            <tr
                                key={entry.id}
                                className="border-b border-gray-100 dark:border-slate-700/60 last:border-0"
                            >
                                <td className="px-4 py-2 whitespace-nowrap">
                                    {entry.created_at?.replace('T', ' ').slice(0, 16) ?? 'Unknown'}
                                </td>
                                <td className="px-4 py-2">
                                    <DriftSeverityBadge severity={entry.severity} />
                                </td>
                                <td className="px-4 py-2">
                                    <DriftStatusBadge status={entry.status} />
                                </td>
                                <td className="px-4 py-2 capitalize">
                                    {entry.evaluation_status.replace('_', ' ')}
                                </td>
                                <td className="px-4 py-2">
                                    {entry.threshold_version != null ? `v${entry.threshold_version}` : '—'}
                                </td>
                                <td className="px-4 py-2">{entry.owner ?? '—'}</td>
                                <td className="px-4 py-2 text-right">
                                    <button
                                        type="button"
                                        onClick={() => onInvestigate(entry.id)}
                                        className="inline-flex items-center gap-1 px-2 py-1 rounded text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-xs font-medium"
                                    >
                                        <Search size={12} aria-hidden="true" />
                                        Investigate
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
