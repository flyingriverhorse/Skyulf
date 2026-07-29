type MetricDict = Record<string, unknown>;

function asMetricDict(value: unknown): MetricDict | null {
  return value && typeof value === 'object' ? (value as MetricDict) : null;
}

function getStepDetails(steps: MetricDict, stepKey: string): MetricDict | null {
  const step = asMetricDict(steps[stepKey]);
  return asMetricDict(step?.details);
}

export function getNodeMetricDetails(
  metrics: unknown,
  options?: { stepKey?: string },
): MetricDict | null {
  const metricDict = asMetricDict(metrics);
  if (!metricDict) return null;

  const steps = asMetricDict(metricDict.steps);
  if (!steps) return metricDict;

  if (options?.stepKey) {
    return getStepDetails(steps, options.stepKey);
  }

  const stepKeys = Object.keys(steps).filter((stepKey) => getStepDetails(steps, stepKey));
  if (stepKeys.length !== 1) return null;

  return getStepDetails(steps, stepKeys[0]!);
}
