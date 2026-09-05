/**
 * Shared bar-chart data preparation for Feature Importance and SHAP Summary,
 * which both compare a per-feature score across selected runs with the same
 * per-run min-max normalization and the same "not reported" semantics. This
 * is a single source of truth so a feature a run doesn't report renders
 * distinctly from a feature genuinely scored at 0 (UX finding EXP-003).
 */
import { shortRunId } from './jobMeta';

export interface FeatureBarJobEntry {
  jobId: string;
  pipeline_id: string;
  parent_pipeline_id?: string | null;
  modelType: string;
  /** Raw (non-normalized) per-feature values as reported by the run, or
   * null if this run doesn't report the artifact at all. */
  values: Record<string, number> | null;
}

/** One bar-chart row: `feature` plus one numeric value per run's bar key,
 * plus a `${barKey}__reported` boolean flag marking whether that run
 * actually reported this feature (vs the 0 being a placeholder). */
export type FeatureBarChartRow = Record<string, string | number | boolean>;

export interface FeatureBarChartData {
  chartData: FeatureBarChartRow[];
  barKeys: string[];
  topFeatures: string[];
  allFeatures: Set<string>;
  jobsWithDataCount: number;
  /** Raw (non-normalized) value lookup, keyed by [barKey][feature], for the
   * data-table alternative — so the table can show the real magnitude
   * alongside the normalized bar value. */
  rawByBarKey: Record<string, Record<string, number>>;
}

/** Suffix appended to a bar key to store whether that run reported the feature. */
export const reportedFlagKey = (barKey: string): string => `${barKey}__reported`;

/**
 * Normalizes a single job's values so the largest magnitude = 1.0. Different
 * model families/output scales (e.g. gini vs gain vs permutation importance,
 * or log-odds vs raw-unit SHAP) are otherwise incomparable side by side.
 */
function normalizePerJob(raw: Record<string, number>): Record<string, number> {
  const max = Math.max(...Object.values(raw).map((v) => Math.abs(v)));
  if (!Number.isFinite(max) || max === 0) return raw;
  const out: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    out[k] = v / max;
  }
  return out;
}

/**
 * Builds the top-15-feature bar chart dataset shared by Feature Importance
 * and SHAP Summary: per-run min-max normalization, average-importance
 * ranking across runs that report a feature, and an explicit "reported"
 * flag per (feature, run) cell so a feature a run doesn't score can be
 * rendered differently from one genuinely scored at 0.
 */
export function buildFeatureBarChartData(jobs: FeatureBarJobEntry[]): FeatureBarChartData {
  const jobsWithData = jobs.filter((j) => j.values !== null);

  const normalized = jobsWithData.map((j) => ({
    ...j,
    raw: j.values ?? {},
    normalized: normalizePerJob(j.values ?? {}),
  }));

  const allFeatures = new Set<string>();
  normalized.forEach((j) => {
    Object.keys(j.normalized).forEach((f) => allFeatures.add(f));
  });

  const featureAvg = Array.from(allFeatures).map((f) => {
    let sum = 0;
    let count = 0;
    normalized.forEach((j) => {
      const val = j.normalized[f];
      if (val !== undefined) {
        sum += Math.abs(val);
        count++;
      }
    });
    return { feature: f, avg: count > 0 ? sum / count : 0 };
  });
  featureAvg.sort((a, b) => b.avg - a.avg);
  const topFeatures = featureAvg.slice(0, 15).map((f) => f.feature);

  const barKeys = normalized.map((j) => {
    const shortId = shortRunId(j);
    return j.modelType !== 'unknown' ? `${j.modelType} (${shortId})` : shortId;
  });

  const rawByBarKey: Record<string, Record<string, number>> = {};
  normalized.forEach((j, i) => {
    rawByBarKey[barKeys[i]!] = j.raw;
  });

  const chartData: FeatureBarChartRow[] = topFeatures.map((feature) => {
    const row: FeatureBarChartRow = { feature };
    normalized.forEach((j, i) => {
      const barKey = barKeys[i]!;
      const reported = j.normalized[feature] !== undefined;
      row[barKey] = j.normalized[feature] ?? 0;
      row[reportedFlagKey(barKey)] = reported;
    });
    return row;
  });

  return { chartData, barKeys, topFeatures, allFeatures, jobsWithDataCount: jobsWithData.length, rawByBarKey };
}
