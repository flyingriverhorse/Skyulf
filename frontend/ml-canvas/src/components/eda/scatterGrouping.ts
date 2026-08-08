import { CHART_SERIES_COLORS } from '../../core/theme/chartTheme';
import { markerShapeForIndex } from './chartMarkerShapes';
import type { ChartLegendEntry } from './ChartLegend';

export type ScatterPointLike = Record<string, string | number | null | undefined>;

/**
 * Groups scatter points by the string value of `labelKey`, falling back to a
 * single "Data Points" group when no `labelKey` is provided. Shared by
 * `CanvasScatterPlot` and `ThreeDScatterPlot` so both chart engines and the
 * accompanying `ChartLegend`/`ChartDataTable` agree on group order and color.
 */
export const groupScatterPoints = (
  data: ScatterPointLike[],
  labelKey?: string | undefined
): Record<string, ScatterPointLike[]> => {
  if (!labelKey) return { 'Data Points': data };

  const groups: Record<string, ScatterPointLike[]> = {};
  data.forEach((point) => {
    const label = String(point[labelKey] ?? 'Other');
    (groups[label] ??= []).push(point);
  });
  return groups;
};

/** Builds legend entries (color + shape) for a set of scatter groups, in the same order the chart renders them. */
export const buildScatterLegendEntries = (groups: Record<string, ScatterPointLike[]>): ChartLegendEntry[] =>
  Object.keys(groups).map((label, idx) => ({
    label,
    color: CHART_SERIES_COLORS[idx % CHART_SERIES_COLORS.length]!,
    shape: markerShapeForIndex(idx),
  }));
