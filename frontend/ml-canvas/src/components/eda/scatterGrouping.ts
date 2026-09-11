import { CHART_SERIES_COLORS } from '../../core/theme/chartTheme';
import { markerShapeForIndex } from './chartMarkerShapes';
import type { ChartLegendEntry } from './ChartLegend';

interface ScatterGroup<T> extends ChartLegendEntry {
  /** Original category identity; null is reserved for missing values. */
  value: string | null;
  points: T[];
}

/** Alphabetize consistently across browser locales, breaking ties between distinct Unicode spellings. */
const compareLabels = (left: string, right: string): number => {
  const alphabetical = left.localeCompare(right, 'en');
  if (alphabetical !== 0 || left === right) return alphabetical;
  return left < right ? -1 : 1;
};

/** Chooses a missing-label caption that cannot impersonate an observed category. */
const missingGroupLabel = (labels: ReadonlySet<string>): string => {
  let label = 'Unlabeled';
  let suffix = 1;
  while (labels.has(label)) {
    label = suffix === 1 ? 'Unlabeled (missing)' : `Unlabeled (missing) ${suffix}`;
    suffix += 1;
  }
  return label;
};

/**
 * Shares category identity and styling across scatter plots, maps and legends.
 * Sorting observed labels stabilizes colors/shapes for the same category set.
 * Missing values keep their own neutral group without shifting category colors.
 */
export const groupScatterPoints = <T extends Record<string, unknown>>(
  data: T[],
  labelKey?: string | undefined
): ScatterGroup<T>[] => {
  if (!labelKey) return [{
    value: null, label: 'Data Points', points: data,
    color: CHART_SERIES_COLORS[0]!, shape: markerShapeForIndex(0),
  }];

  const groups = new Map<string | null, T[]>();
  data.forEach((point) => {
    const value = point[labelKey];
    const label = value == null ? null : String(value);
    const points = groups.get(label) ?? [];
    points.push(point);
    groups.set(label, points);
  });

  const labels = [...groups.keys()].filter((label): label is string => label !== null).sort(compareLabels);
  const result: ScatterGroup<T>[] = labels.map((label, idx) => ({
    value: label, label, points: groups.get(label)!,
    color: CHART_SERIES_COLORS[idx % CHART_SERIES_COLORS.length]!,
    shape: markerShapeForIndex(idx),
  }));
  const missing = groups.get(null);
  if (missing) result.push({
    value: null, label: missingGroupLabel(new Set(labels)), points: missing,
    color: '#6b7280', shape: 'cross',
  });
  return result;
};

/** Builds legend entries (color + shape) for a set of scatter groups, in the same order the chart renders them. */
export const buildScatterLegendEntries = (groups: readonly ChartLegendEntry[]): ChartLegendEntry[] =>
  groups.map(({ label, color, shape }) => ({ label, color, shape }));
