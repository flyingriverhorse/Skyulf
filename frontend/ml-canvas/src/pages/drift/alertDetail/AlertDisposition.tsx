import { ShieldCheck } from 'lucide-react';
import type { DriftAlertDetail, DriftDispositionAction } from '../../../core/api/monitoring';
import { FormField } from '../../../components/ui/FormField';
import { AlertHistory } from './AlertHistory';

/** Actions available from each disposition status, in display order. */
const NEXT_ACTIONS: Record<string, DriftDispositionAction[]> = {
    new: ['acknowledge'],
    acknowledged: ['resolve', 'reopen'],
    resolved: ['reopen'],
    reopened: ['acknowledge'],
};

const ACTION_LABELS: Record<DriftDispositionAction, string> = {
    acknowledge: 'Acknowledge',
    resolve: 'Resolve',
    reopen: 'Reopen',
};

interface AlertDispositionProps {
    detail: DriftAlertDetail;
    actor: string;
    note: string;
    actionError: string | null;
    error: string | null;
    actionPending: boolean;
    setActor: (actor: string) => void;
    setNote: (note: string) => void;
    handleAction: (action: DriftDispositionAction) => Promise<void>;
}

/** Render controlled audit fields without taking ownership of their lifetime. */
export function AlertDisposition({
    detail, actor, note, actionError, error, actionPending, setActor, setNote, handleAction,
}: AlertDispositionProps) {
    return (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1.5">
                <ShieldCheck size={13} /> Disposition
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {detail.owner ? (
                    <>
                        Currently <strong>{detail.status}</strong> by <strong>{detail.owner}</strong>.
                    </>
                ) : (
                    <>No disposition recorded yet — acknowledge to claim this alert.</>
                )}
            </p>

            {(NEXT_ACTIONS[detail.status] ?? []).length > 0 && (
                <div className="space-y-2 mb-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        <FormField label="Your name" required>
                            {field => (
                                <input
                                    {...field}
                                    type="text"
                                    value={actor}
                                    onChange={e => setActor(e.target.value)}
                                    placeholder="e.g. alice"
                                    className="w-full text-sm px-3 py-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800"
                                />
                            )}
                        </FormField>
                        <FormField label="Note" hint="Optional context for the audit trail">
                            {field => (
                                <input
                                    {...field}
                                    type="text"
                                    value={note}
                                    onChange={e => setNote(e.target.value)}
                                    placeholder="Optional"
                                    className="w-full text-sm px-3 py-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800"
                                />
                            )}
                        </FormField>
                    </div>
                    {actionError && (
                        <p className="text-xs text-red-600 dark:text-red-400">{actionError}</p>
                    )}
                    {error && <p className="text-xs text-red-600 dark:text-red-400">{error}</p>}
                    <div className="flex gap-2">
                        {(NEXT_ACTIONS[detail.status] ?? []).map(action => (
                            <button
                                key={action}
                                type="button"
                                disabled={actionPending}
                                onClick={() => void handleAction(action)}
                                className="px-3 py-1.5 text-sm font-medium rounded-md action-primary disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {ACTION_LABELS[action]}
                            </button>
                        ))}
                    </div>
                </div>
            )}
            <AlertHistory entries={detail.disposition_history} />
        </div>
    );
}
