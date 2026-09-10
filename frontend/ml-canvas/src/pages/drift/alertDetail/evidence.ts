import type { DriftAlertDetail } from '../../../core/api/monitoring';

/** Preserve retained feature order and round the last value for each metric. */
export function prepareEvidenceRows(detail: DriftAlertDetail | null) {
    return detail?.column_drifts
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
                  ks_statistic:
                      metrics.ks_statistic != null ? Number(metrics.ks_statistic.toFixed(4)) : null,
                  ks_p_value:
                      metrics.ks_test_p_value != null ? Number(metrics.ks_test_p_value.toFixed(4)) : null,
              };
          })
        : [];
}
