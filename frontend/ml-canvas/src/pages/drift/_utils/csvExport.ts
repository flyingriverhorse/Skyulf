import type { DriftReport } from '../../../core/api/monitoring';

function getRisk(
    featureImportances: DriftReport['feature_importances'],
    importance: number | undefined,
    driftDetected: boolean,
    rank: number | null,
): string {
    if (featureImportances && importance == null) return 'Unknown';
    if (driftDetected && rank != null) {
        if (rank <= 5) return 'High';
        return rank <= 15 ? 'Medium' : 'Low';
    }
    return featureImportances ? 'Low' : '';
}

/**
 * Serialise the (threshold-evaluated) drift report to CSV and trigger a
 * browser download. Includes per-feature importance and risk class when the
 * backend provided feature_importances.
 */
export function exportDriftReportCSV(report: DriftReport, datasetName: string | undefined): void {
    const fi = report.feature_importances;
    const headers = [
        'Column',
        'Status',
        'Wasserstein',
        'PSI',
        'KL Divergence',
        'KS Statistic',
        'KS P-Value',
        ...(fi ? ['Importance', 'Risk'] : []),
    ];
    const rows = Object.values(report.column_drifts).map(col => {
        const get = (m: string) => col.metrics.find(
            x => x.metric === m || (m === 'psi' && x.metric === 'psi_categorical'),
        )?.value?.toFixed(6) ?? '';
        const importance = fi?.[col.column];
        const rank = fi && importance != null ? Object.values(fi).filter(v => v > importance).length + 1 : null;
        const risk = getRisk(fi, importance, col.drift_detected, rank);
        return [
            col.column,
            col.drift_detected ? 'Drifted' : 'Stable',
            get('wasserstein_distance'),
            get('psi'),
            get('kl_divergence'),
            get('ks_statistic'),
            get('ks_test_p_value'),
            ...(fi ? [importance?.toFixed(6) ?? '', risk] : []),
        ];
    });
    const csv = [headers, ...rows].map(r => r.map(c => `"${c.replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drift_report_${datasetName ?? 'export'}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}
