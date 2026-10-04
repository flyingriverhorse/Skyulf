type TemporalConfig = {
  cv_type?: unknown;
  cv_gap?: unknown;
  cv_test_size?: unknown;
  cv_max_train_size?: unknown;
};

/** Temporal CV with explicit windows or nested folds requires named timestamp metadata. */
export function requiresExplicitCVTimeColumn(config: TemporalConfig): boolean {
  return config.cv_type === 'nested_cv' || [config.cv_gap, config.cv_test_size, config.cv_max_train_size].some(Boolean);
}
