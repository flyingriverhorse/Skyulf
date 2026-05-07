import type { DriftReport } from '../../../core/api/monitoring';

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
        'KS P-Value',
        ...(fi ? ['Importance', 'Risk'] : []),
    ];
    const rows = Object.values(report.column_drifts).map(col => {
        const get = (m: string) => col.metrics.find(x => x.metric === m)?.value?.toFixed(6) ?? '';
        const importance = fi?.[col.column];
        const rank = fi ? Object.values(fi).filter(v => v > (importance ?? 0)).length + 1 : null;
        const risk =
            col.drift_detected && rank != null
                ? rank <= 5
                    ? 'High'
                    : rank <= 15
                    ? 'Medium'
                    : 'Low'
                : fi
                ? 'Low'
                : '';
        return [
            col.column,
            col.drift_detected ? 'Drifted' : 'Stable',
            get('wasserstein_distance'),
            get('psi'),
            get('kl_divergence'),
            get('ks_test_p_value'),
            ...(fi ? [importance?.toFixed(6) ?? '', risk] : []),
        ];
    });
    const csv = [headers, ...rows].map(r => r.map(c => `"${c}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `drift_report_${datasetName ?? 'export'}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}
