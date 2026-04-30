// Family-aware perf-overlay thresholds.
//
// A flat threshold paints every trainer red and conveys no signal,
// because a "successful" trainer can take anywhere from a few hundred
// milliseconds (logistic regression on 1k rows) to several minutes
// (XGBoost + 5-fold CV + 100-trial HPO). We split nodes into three
// families and pick bands that match the reality of each.
//
// Numbers chosen from sampled user runs:
//   - Preprocess: pandas/sklearn transformers. Most finish in <500 ms,
//     but heavy ones (TF-IDF on 1M docs, large encoders, big merges)
//     legitimately take seconds. ≥5 s is the "consider sampling /
//     caching" zone.
//   - Basic training: a single `.fit` call on the full dataset. Modern
//     CPUs handle classic sklearn fits in seconds; ≥60 s typically
//     means the dataset is large enough that you should consider
//     subsampling for iteration.
//   - Tuner / HPO: Optuna search with cross-validation (50 trials × 5
//     folds = 250 fits) is *expected* to take minutes. Anything under
//     a minute is suspiciously cheap; ≥10 min is where you start
//     thinking "fewer trials, smaller search space, or time-budget".

export type PerfFamily = 'preprocess' | 'trainer' | 'tuner';
export type PerfBucket = 'fast' | 'medium' | 'slow';

export interface PerfBands {
  /** Upper bound (exclusive) for the green band, in ms. */
  fastMaxMs: number;
  /** Upper bound (exclusive) for the amber band, in ms. */
  mediumMaxMs: number;
}

const BANDS: Record<PerfFamily, PerfBands> = {
  preprocess: { fastMaxMs: 500, mediumMaxMs: 5_000 },
  trainer: { fastMaxMs: 5_000, mediumMaxMs: 60_000 },
  tuner: { fastMaxMs: 60_000, mediumMaxMs: 600_000 },
};

export function getPerfFamily(definitionType: string): PerfFamily {
  if (definitionType === 'advanced_tuning') return 'tuner';
  if (definitionType === 'basic_training') return 'trainer';
  return 'preprocess';
}

export function getPerfBands(family: PerfFamily): PerfBands {
  return BANDS[family];
}

export function bucketDuration(durationMs: number, family: PerfFamily): PerfBucket {
  const bands = BANDS[family];
  if (durationMs < bands.fastMaxMs) return 'fast';
  if (durationMs < bands.mediumMaxMs) return 'medium';
  return 'slow';
}

/** Human-readable like "<500 ms", "<5 s", "≥10 min". */
export function formatBoundMs(ms: number, kind: '<' | '>='): string {
  const prefix = kind === '<' ? '<' : '≥';
  if (ms < 1_000) return `${prefix}${ms} ms`;
  if (ms < 60_000) return `${prefix}${Math.round(ms / 100) / 10} s`.replace('.0 s', ' s');
  return `${prefix}${Math.round(ms / 60_000)} min`;
}

export interface PerfFamilyDescriptor {
  family: PerfFamily;
  label: string;
  /** Pre-rendered short string e.g. "<500 ms · <5 s · ≥5 s". */
  bandsLabel: string;
}

export const PERF_FAMILY_DESCRIPTORS: PerfFamilyDescriptor[] = (
  ['preprocess', 'trainer', 'tuner'] as const
).map((family) => {
  const b = BANDS[family];
  const labels: Record<PerfFamily, string> = {
    preprocess: 'Preprocess',
    trainer: 'Trainer',
    tuner: 'Tuner / HPO',
  };
  return {
    family,
    label: labels[family],
    bandsLabel: `${formatBoundMs(b.fastMaxMs, '<')} · ${formatBoundMs(b.mediumMaxMs, '<')} · ${formatBoundMs(b.mediumMaxMs, '>=')}`,
  };
});
