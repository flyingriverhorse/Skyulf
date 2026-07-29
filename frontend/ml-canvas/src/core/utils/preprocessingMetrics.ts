type MetricDict = Record<string, unknown>;

function asMetricDict(value: unknown): MetricDict | null {
  return value && typeof value === 'object' ? (value as MetricDict) : null;
}

export function getNodeMetricDetails(metrics: unknown): MetricDict | null {
  const metricDict = asMetricDict(metrics);
  if (!metricDict) return null;

  const steps = asMetricDict(metricDict.steps);
  if (!steps) return metricDict;

  for (const key of Object.keys(steps).sort()) {
    const step = asMetricDict(steps[key]);
    const details = asMetricDict(step?.details);
    if (details) return details;
  }

  return metricDict;
}
