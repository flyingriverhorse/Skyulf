/** Validate scaled thresholds without changing estimator class order or valid binary boundaries. */
export function validateThresholds(
  thresholds: Record<string, number>,
  classes: readonly (string | number)[] = Object.keys(thresholds),
): void {
  const values = Object.values(thresholds);
  if (classes.length > 2) {
    if (values.length !== classes.length || classes.some(cls => !Object.prototype.hasOwnProperty.call(thresholds, String(cls)))) {
      throw new Error('Multiclass thresholds must cover every class exactly once.');
    }
    if (values.some(value => !Number.isFinite(value) || value <= 0)) {
      throw new Error('Multiclass thresholds must be finite numbers greater than 0.');
    }
  } else if (values.some(value => !Number.isFinite(value) || value < 0)) {
    throw new Error('Binary thresholds must be finite, non-negative numbers.');
  } else if (values.length === 2 && values.every(value => value === 0)) {
    throw new Error('Binary thresholds cannot both be 0.');
  }
}
