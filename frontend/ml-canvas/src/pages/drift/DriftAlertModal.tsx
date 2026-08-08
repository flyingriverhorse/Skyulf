import React, { useState } from 'react';
import { AlertCircle, Clock, History, ShieldCheck, User } from 'lucide-react';
import type { DriftAlertDetail, DriftDispositionAction } from '../../core/api/monitoring';
import { EmptyState, ErrorState, LoadingState, ModalShell, RecordLink } from '../../components/shared';
import { FormField } from '../../components/ui/FormField';
import { ChartDataTable } from '../../components/eda/ChartDataTable';
import { DriftSeverityBadge, DriftStatusBadge } from './DriftAlertBadges';

interface DriftAlertModalProps {
    alertId: number | null;
    detail: DriftAlertDetail | null;
    loading: boolean;
    error: string | null;
    actionPending: boolean;
    onApplyDisposition: (
        action: DriftDispositionAction,
        actor: string,
        note?: string,
    ) => Promise<unknown>;
    onRetry: () => void;
    onClose: () => void;
    filters: Record<string, string>;
}

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

/**
 * OPS-003 investigation surface for a single drift alert: identity, severity,
 * the threshold version it was evaluated against, per-feature evidence, links
 * to the related job/model version/deployment, and the acknowledge / resolve
 * / reopen disposition workflow with its full actor/timestamp audit trail.
 */
export const DriftAlertModal: React.FC<DriftAlertModalProps> = ({
    alertId,
    detail,
    loading,
    error,
    actionPending,
    onApplyDisposition,
    onRetry,
    onClose,
    filters,
}) => {
    const [actor, setActor] = useState('');
    const [note, setNote] = useState('');
    const [actionError, setActionError] = useState<string | null>(null);

    const handleAction = async (action: DriftDispositionAction) => {
        if (!actor.trim()) {
            setActionError('Enter your name so the disposition records who made it.');
            return;
        }
        setActionError(null);
        const result = await onApplyDisposition(action, actor.trim(), note.trim() || undefined);
        if (result) setNote('');
    };

    const evidenceRows = detail?.column_drifts
        ? Object.entries(detail.column_drifts).map(([column, drift]) => {
              const metrics = Object.fromEntries(drift.metrics.map(m => [m.metric, m.value]));
              return {
                  column,
                  drifted: drift.drift_detected ? 'Yes' : 'No',
                  psi: metrics.psi != null ? Number(metrics.psi.toFixed(4)) : null,
                  wasserstein:
                      metrics.wasserstein_distance != null
                          ? Number(metrics.wasserstein_distance.toFixed(4))
                          : null,
                  ks_p_value:
                      metrics.ks_test_p_value != null ? Number(metrics.ks_test_p_value.toFixed(4)) : null,
              };
          })
        : [];

    return (
        <ModalShell
            isOpen={alertId !== null}
            onClose={onClose}
            title={alertId !== null ? `Drift alert #${alertId}` : undefined}
            size="3xl"
        >
            {loading && <LoadingState message="Loading drift alert…" />}
            {!loading && error && !detail && <ErrorState error={error} onRetry={onRetry} />}
            {!loading && detail && (
                <div className="p-6 space-y-5">
                    <div className="flex flex-wrap items-center gap-2">
                        <DriftSeverityBadge severity={detail.severity} />
                        <DriftStatusBadge status={detail.status} />
                        {detail.evaluation_status !== 'completed' && (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700">
                                <AlertCircle size={12} aria-hidden="true" />
                                {detail.evaluation_status === 'no_baseline' ? 'No baseline' : 'Evaluation failed'}
                            </span>
                        )}
                    </div>

                    {detail.evaluation_status !== 'completed' && detail.error_message && (
                        <div className="flex items-start gap-2 p-3 rounded border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300 text-sm">
                            <AlertCircle size={16} className="shrink-0 mt-0.5" aria-hidden="true" />
                            <span>{detail.error_message}</span>
                        </div>
                    )}

                    <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                        <div>
                            <dt className="text-xs text-gray-500 dark:text-gray-400">Detected</dt>
                            <dd className="font-medium">{detail.created_at?.replace('T', ' ').slice(0, 16) ?? 'Unknown'}</dd>
                        </div>
                        <div>
                            <dt className="text-xs text-gray-500 dark:text-gray-400">Threshold version</dt>
                            <dd className="font-medium">
                                {detail.threshold_version != null ? `v${detail.threshold_version}` : '—'}
                                {detail.threshold_version != null && (
                                    <span className="block text-[11px] font-normal text-gray-400">
                                        PSI {detail.threshold_psi} · KS {detail.threshold_ks} · Wasserstein{' '}
                                        {detail.threshold_wasserstein} · KL {detail.threshold_kl}
                                    </span>
                                )}
                            </dd>
                        </div>
                        <div>
                            <dt className="text-xs text-gray-500 dark:text-gray-400">Drifted features</dt>
                            <dd className="font-medium">
                                {detail.drifted_columns_count != null && detail.total_columns != null
                                    ? `${detail.drifted_columns_count} / ${detail.total_columns}`
                                    : '—'}
                            </dd>
                        </div>
                    </dl>

                    <div className="flex flex-wrap items-center gap-3 text-sm">
                        <RecordLink
                            recordRef={{ kind: 'job', jobId: detail.job_id }}
                            origin="/drift"
                            filters={filters}
                        />
                        {detail.model_version && (
                            <RecordLink
                                recordRef={{
                                    kind: 'modelVersion',
                                    jobId: detail.job_id,
                                    version: detail.model_version,
                                }}
                                origin="/drift"
                                filters={filters}
                            />
                        )}
                        {detail.deployment_id != null && (
                            <RecordLink
                                recordRef={{ kind: 'deployment', deploymentId: detail.deployment_id }}
                                origin="/drift"
                                filters={filters}
                            />
                        )}
                    </div>

                    {detail.evaluation_status === 'completed' && (
                        <div>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                                Feature evidence
                            </h3>
                            {evidenceRows.length === 0 ? (
                                <EmptyState
                                    title="No evidence recorded"
                                    description="This alert has no retained per-feature drift evidence."
                                />
                            ) : (
                                <ChartDataTable
                                    caption={`Per-feature drift evidence for alert #${alertId ?? ''}`}
                                    filename={`drift-alert-${alertId ?? 'unknown'}-evidence`}
                                    columns={[
                                        { key: 'column', label: 'Feature' },
                                        { key: 'drifted', label: 'Drifted' },
                                        { key: 'psi', label: 'PSI' },
                                        { key: 'wasserstein', label: 'Wasserstein' },
                                        { key: 'ks_p_value', label: 'KS p-value' },
                                    ]}
                                    rows={evidenceRows}
                                    defaultOpen
                                />
                            )}
                        </div>
                    )}

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
                                            className="px-3 py-1.5 text-sm font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            {ACTION_LABELS[action]}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2 flex items-center gap-1.5">
                            <History size={12} /> History
                        </h4>
                        {detail.disposition_history.length === 0 ? (
                            <p className="text-xs text-gray-400 italic">No disposition changes recorded yet.</p>
                        ) : (
                            <ul className="space-y-1.5">
                                {detail.disposition_history.map((entry, idx) => (
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
                    </div>
                </div>
            )}
        </ModalShell>
    );
};
