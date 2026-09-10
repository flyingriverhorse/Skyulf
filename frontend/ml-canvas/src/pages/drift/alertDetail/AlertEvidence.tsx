import type { DriftAlertDetail } from '../../../core/api/monitoring';
import { EmptyState } from '../../../components/shared';
import { ChartDataTable } from '../../../components/eda/ChartDataTable';
import type { prepareEvidenceRows } from './evidence';

interface AlertEvidenceProps {
    alertId: number | null;
    evaluationStatus: DriftAlertDetail['evaluation_status'];
    evidenceRows: ReturnType<typeof prepareEvidenceRows>;
}

/** Show the retained evidence table only for completed evaluations. */
export function AlertEvidence({ alertId, evaluationStatus, evidenceRows }: AlertEvidenceProps) {
    if (evaluationStatus !== 'completed') return null;
    return (
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
                        { key: 'ks_statistic', label: 'KS statistic' },
                        { key: 'ks_p_value', label: 'KS p-value' },
                    ]}
                    rows={evidenceRows}
                    defaultOpen
                />
            )}
        </div>
    );
}
