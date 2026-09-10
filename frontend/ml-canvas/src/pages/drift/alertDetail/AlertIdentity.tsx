import { AlertCircle } from 'lucide-react';
import type { DriftAlertDetail } from '../../../core/api/monitoring';
import { RecordLink } from '../../../components/shared';
import { DriftSeverityBadge, DriftStatusBadge } from '../DriftAlertBadges';

interface AlertIdentityProps {
    detail: DriftAlertDetail;
    filters: Record<string, string>;
}

/** Display the evaluation outcome and its pinned investigation context. */
export function AlertIdentity({ detail, filters }: AlertIdentityProps) {
    return (
        <>
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

            <AlertMetadata detail={detail} />
            <AlertContextLinks detail={detail} filters={filters} />
        </>
    );
}

/** Keep nullish timestamps, thresholds and feature counts distinct from zero. */
function AlertMetadata({ detail }: Pick<AlertIdentityProps, 'detail'>) {
    return (
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
    );
}

/** Preserve related-record ordering and the filters used to return to drift. */
function AlertContextLinks({ detail, filters }: AlertIdentityProps) {
    return (
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
    );
}
