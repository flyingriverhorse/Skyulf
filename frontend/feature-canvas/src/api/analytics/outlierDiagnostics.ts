// @ts-nocheck
import {
  OutlierDiagnosticsResponse,
  FetchOutlierDiagnosticsOptions,
  ManualOutlierBounds,
} from '../types';

export async function fetchOutlierDiagnostics(
  sourceId: string,
  options: FetchOutlierDiagnosticsOptions = {}
): Promise<OutlierDiagnosticsResponse> {
  const { config, sampleSize, graph, targetNodeId } = options;

  const payload: Record<string, any> = {
    dataset_source_id: sourceId,
    sample_size:
      sampleSize === 'all'
        ? 0
        : typeof sampleSize === 'number' && Number.isFinite(sampleSize)
          ? sampleSize
          : 300,
  };

  if (config && typeof config === 'object') {
    const clampPercentile = (value: any, fallback: number) => {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) {
        return fallback;
      }
      if (numeric < 0) {
        return 0;
      }
      if (numeric > 100) {
        return 100;
      }
      return numeric;
    };

    const sanitizeManualBounds = (raw: any): Record<string, ManualOutlierBounds> => {
      if (!raw || typeof raw !== 'object') {
        return {};
      }
      const result: Record<string, ManualOutlierBounds> = {};
      Object.entries(raw as Record<string, any>).forEach(([key, value]) => {
        const column = String(key ?? '').trim();
        if (!column) {
          return;
        }

        let lower: number | null = null;
        let upper: number | null = null;

        if (value && typeof value === 'object') {
          const lowerCandidate = (value as any).lower ?? (value as any).min ?? (value as any).minimum ?? null;
          const upperCandidate = (value as any).upper ?? (value as any).max ?? (value as any).maximum ?? null;

          const parsedLower = Number(lowerCandidate);
          if (Number.isFinite(parsedLower)) {
            lower = parsedLower;
          }

          const parsedUpper = Number(upperCandidate);
          if (Number.isFinite(parsedUpper)) {
            upper = parsedUpper;
          }
        }

        if (lower === null && upper === null) {
          return;
        }

        if (lower !== null && upper !== null && lower > upper) {
          const temp = lower;
          lower = upper;
          upper = temp;
        }

        result[column] = { lower, upper };
      });
      return result;
    };

    const lowerPercentile = clampPercentile((config as any).lower_percentile, 5);
    const upperPercentile = clampPercentile((config as any).upper_percentile, 95);
    payload.config = {
      ...config,
      method: config.method ?? 'z_score',
      columns: Array.isArray(config.columns) ? config.columns : [],
      z_threshold:
        typeof config.z_threshold === 'number' && Number.isFinite(config.z_threshold)
          ? config.z_threshold
          : 3,
      iqr_multiplier:
        typeof config.iqr_multiplier === 'number' && Number.isFinite(config.iqr_multiplier)
          ? config.iqr_multiplier
          : 1.5,
      lower_percentile: lowerPercentile,
      upper_percentile: upperPercentile <= lowerPercentile ? Math.min(lowerPercentile + 1, 100) : upperPercentile,
      manual_bounds: sanitizeManualBounds((config as any).manual_bounds),
    };
  }

  if (graph && Array.isArray(graph.nodes) && Array.isArray(graph.edges)) {
    payload.graph = {
      nodes: graph.nodes,
      edges: graph.edges,
    };
  }

  if (targetNodeId) {
    payload.target_node_id = targetNodeId;
  }

  const response = await fetch('/ml-workflow/api/analytics/outliers', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to load outlier diagnostics');
  }

  return response.json();
}
