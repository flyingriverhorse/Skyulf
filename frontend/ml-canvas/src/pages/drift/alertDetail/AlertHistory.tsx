import { Clock, History, User } from 'lucide-react';
import type { DriftDispositionEntry } from '../../../core/api/monitoring';
import { DriftStatusBadge } from '../DriftAlertBadges';

/** Render the disposition audit trail in the order supplied by the API. */
export function AlertHistory({ entries }: { entries: DriftDispositionEntry[] }) {
    return (
        <>
            <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1.5">
                <History size={12} /> History
            </h4>
            {entries.length === 0 ? (
                <p className="text-xs text-gray-400 italic">No disposition changes recorded yet.</p>
            ) : (
                <ul className="space-y-1.5">
                    {entries.map((entry, idx) => (
                        <li
                            key={idx}
                            className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-600 dark:text-gray-300"
                        >
                            <DriftStatusBadge status={entry.status} />
                            <span className="inline-flex items-center gap-1">
                                <User size={11} /> {entry.actor}
                            </span>
                            <span className="inline-flex items-center gap-1 text-gray-400">
                                <Clock size={11} /> {entry.at.replace('T', ' ').slice(0, 16)}
                            </span>
                            {entry.note && <span className="italic">&quot;{entry.note}&quot;</span>}
                        </li>
                    ))}
                </ul>
            )}
        </>
    );
}
