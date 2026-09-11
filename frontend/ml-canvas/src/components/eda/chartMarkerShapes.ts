/**
 * Shared marker-shape cycle used so grouped scatter charts distinguish
 * groups by shape as well as by color (UX finding DAT-007: color must never
 * be the sole carrier of group identity). Each chart adapter (Chart.js,
 * Plotly scatter3d) maps this shared order onto its own shape vocabulary.
 * Only shapes supported by both renderers belong here: scatter3d silently
 * coerces unsupported symbols such as triangle-up and star to circles.
 */
export const CHART_MARKER_SHAPES = [
  'circle',
  'square',
  'diamond',
  'cross',
  'x',
] as const;

export type ChartMarkerShape = (typeof CHART_MARKER_SHAPES)[number];

/** Chart.js `pointStyle` value for a given shared shape. */
const CHARTJS_POINT_STYLES: Record<ChartMarkerShape, string> = {
  circle: 'circle',
  square: 'rect',
  diamond: 'rectRot',
  cross: 'cross',
  x: 'crossRot',
};

/** Plotly scatter3d `marker.symbol` value for a given shared shape. */
const PLOTLY_MARKER_SYMBOLS: Record<ChartMarkerShape, string> = {
  circle: 'circle',
  square: 'square',
  diamond: 'diamond',
  cross: 'cross',
  x: 'x',
};

/** Picks a shape from the shared cycle for the group at `index`. */
export const markerShapeForIndex = (index: number): ChartMarkerShape =>
  CHART_MARKER_SHAPES[index % CHART_MARKER_SHAPES.length]!;

/** Maps a shared shape name to the Chart.js `pointStyle` string. */
export const toChartJsPointStyle = (shape: ChartMarkerShape): string => CHARTJS_POINT_STYLES[shape];

/** Maps a shared shape name to the Plotly `marker.symbol` string. */
export const toPlotlyMarkerSymbol = (shape: ChartMarkerShape): string => PLOTLY_MARKER_SYMBOLS[shape];
