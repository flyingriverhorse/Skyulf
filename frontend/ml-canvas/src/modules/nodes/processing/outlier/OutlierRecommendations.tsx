import type { OutlierConfig } from './types';
import type { Recommendation } from '../../../../core/api/client';
import { RecommendationsPanel } from '../../../../components/panels/RecommendationsPanel';

function getMetricPrefix(method: OutlierConfig['method']): string {
  switch (method) {
    case 'elliptic_envelope': return 'EllipticEnvelope';
    case 'zscore': return 'ZScore';
    case 'winsorize': return 'Winsorize';
    default: return 'IQR';
  }
}

function getLossRecommendations(config: OutlierConfig, metrics: Record<string, unknown>): Recommendation[] {
  const runtimeRecommendations: Recommendation[] = [];
  const rowsRemoved = metrics.rows_removed as number | undefined;
  const rowsTotal = metrics.rows_total as number | undefined;

  // If rowsTotal is missing but we have rowsRemoved, we can still give some feedback
  const effectiveRowsTotal = rowsTotal ?? (metrics[`${getMetricPrefix(config.method)}_rows_total`] as number | undefined);

  if (rowsRemoved !== undefined && effectiveRowsTotal !== undefined && effectiveRowsTotal > 0) {
    const lossRatio = rowsRemoved / effectiveRowsTotal;
    if (lossRatio > 0.2) {
      runtimeRecommendations.push({
        rule_id: 'high_data_loss',
        type: 'warning',
        description: "High data loss (>20%).",
        reasoning: "Consider relaxing parameters (e.g., higher IQR multiplier or Z-Score threshold) or using Winsorization to clip values instead of removing rows.",
        suggested_node_type: 'OutlierRemoval',
        suggested_params: {},
        confidence: 1.0,
        target_columns: []
      });
    } else if (lossRatio === 0) {
      if (config.method !== 'winsorize') {
        runtimeRecommendations.push({
          rule_id: 'no_outliers',
          type: 'info',
          description: "No outliers detected.",
          reasoning: "If you suspect outliers, try tightening the parameters (e.g., lower IQR multiplier).",
          suggested_node_type: 'OutlierRemoval',
          suggested_params: {},
          confidence: 1.0,
          target_columns: []
        });
      }
    }
  }

  return runtimeRecommendations;
}

export function OutlierRecommendations({ config, metrics, hasNodeResult, backendRecommendations }: {
  config: OutlierConfig;
  metrics: Record<string, unknown> | null;
  hasNodeResult: boolean;
  backendRecommendations: Recommendation[];
}) {
  // Combine backend recommendations with runtime feedback
  const runtimeRecommendations: Recommendation[] = [];

  if (hasNodeResult && metrics) {
    runtimeRecommendations.push(...getLossRecommendations(config, metrics));

    if (config.method === 'elliptic_envelope') {
      runtimeRecommendations.push({
        rule_id: 'elliptic_stochastic',
        type: 'info',
        description: "Stochastic Method",
        reasoning: "Elliptic Envelope is stochastic. Results may vary slightly between runs.",
        suggested_node_type: 'OutlierRemoval',
        suggested_params: {},
        confidence: 1.0,
        target_columns: []
      });
    }
  }

  const allRecommendations = [...backendRecommendations, ...runtimeRecommendations];

  if (allRecommendations.length === 0) return null;

  return (
    <div className="mt-4">
      <RecommendationsPanel recommendations={allRecommendations} />
    </div>
  );
}
