/**
 * Shared marker-shape cycle used so grouped scatter charts distinguish
 * groups by shape as well as by color (UX finding DAT-007: color must never
 * be the sole carrier of group identity). Each chart adapter (Chart.js,
 * Plotly) maps this shared order onto its own shape vocabulary so a given
 * group index renders a consistent shape across chart families.
 */
export const CHART_MARKER_SHAPES = [
  'circle',
  'triangle',
  'square',
  'diamond',
  'star',
  'cross',
] as const;

export type ChartMarkerShape = (typeof CHART_MARKER_SHAPES)[number];

/** Chart.js `pointStyle` value for a given shared shape. */
const CHARTJS_POINT_STYLES: Record<ChartMarkerShape, string> = {
  circle: 'circle',
  triangle: 'triangle',
  square: 'rect',
  diamond: 'rectRot',
  star: 'star',
  cross: 'cross',
};

/** Plotly `marker.symbol` value for a given shared shape. */
const PLOTLY_MARKER_SYMBOLS: Record<ChartMarkerShape, string> = {
  circle: 'circle',
  triangle: 'triangle-up',
  square: 'square',
  diamond: 'diamond',
  star: 'star',
  cross: 'cross',
};

/** Picks a shape from the shared cycle for the group at `index`. */
export const markerShapeForIndex = (index: number): ChartMarkerShape =>
  CHART_MARKER_SHAPES[index % CHART_MARKER_SHAPES.length]!;

/** Maps a shared shape name to the Chart.js `pointStyle` string. */
export const toChartJsPointStyle = (shape: ChartMarkerShape): string => CHARTJS_POINT_STYLES[shape];

/** Maps a shared shape name to the Plotly `marker.symbol` string. */
export const toPlotlyMarkerSymbol = (shape: ChartMarkerShape): string => PLOTLY_MARKER_SYMBOLS[shape];
