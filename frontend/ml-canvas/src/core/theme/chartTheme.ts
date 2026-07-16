/**
 * Single source of truth for chart colors that work on both light and dark
 * backgrounds. Centralized here so every chart (Dashboard, EDA, SHAP,
 * Feature Importance, Evaluation, ...) picks colors from one place instead
 * of each component hardcoding its own gray/black.
 *
 * Two consumers:
 * - `useChartTheme()` (in `../hooks/useChartTheme`) — reactive, for JSX/Recharts.
 * - `getChartTheme()`/`getTooltipContentStyle()` below — imperative snapshot,
 *   for one-off work outside React render (e.g. canvas-based chart PNG export).
 */

export interface ChartColors {
  /** Axis tick/line color. */
  axisColor: string;
  /** CartesianGrid line color. */
  gridColor: string;
  /** Primary text (titles, canvas-drawn headings). */
  textColor: string;
  /** Secondary/muted text (subtitles, captions). */
  subTextColor: string;
  /** Chart/canvas background. */
  bgColor: string;
  /** Ready-to-spread `contentStyle` for a Recharts `<Tooltip>`. */
  tooltipContentStyle: Record<string, string>;
  /** Ready-to-spread `itemStyle` for a Recharts `<Tooltip>` — Recharts
   * defaults each item's text color to `entry.color || '#000'`, and Pie
   * chart payload entries don't carry a usable `color`, so without this the
   * tooltip text renders black-on-dark in dark mode. */
  tooltipItemStyle: Record<string, string>;
  /** Ready-to-spread `labelStyle` for a Recharts `<Tooltip>`, matching the
   * content text color for the (optional) label row above the items. */
  tooltipLabelStyle: Record<string, string>;
}

/** Categorical series palette — mid-saturation colors with enough contrast
 * against both white and gray-900 backgrounds. Safe to use regardless of
 * theme (unlike axis/grid/text colors, which must adapt). */
export const CHART_SERIES_COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe',
  '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57',
];

/** Reads the current theme directly from the DOM. Non-reactive — prefer the
 * `useIsDarkMode()`/`useChartTheme()` hooks inside React components. */
export const isDarkModeActive = (): boolean =>
  typeof document !== 'undefined' && document.documentElement.classList.contains('dark');

export const resolveChartColors = (isDark: boolean): ChartColors => {
  const tooltipTextColor = isDark ? '#f3f4f6' : '#111827';
  return {
    axisColor: isDark ? '#9ca3af' : '#6b7280',       // gray-400 / gray-500
    gridColor: isDark ? 'rgba(255,255,255,0.12)' : 'rgba(0,0,0,0.1)',
    textColor: isDark ? '#e5e7eb' : '#374151',        // gray-200 / gray-700
    subTextColor: isDark ? '#9ca3af' : '#4b5563',      // gray-400 / gray-600
    bgColor: isDark ? '#1f2937' : '#ffffff',           // gray-800 / white
    tooltipContentStyle: isDark
      ? { backgroundColor: '#1f2937', borderRadius: '8px', border: '1px solid #374151', color: tooltipTextColor, boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)' }
      : { backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '8px', border: '1px solid #e5e7eb', color: tooltipTextColor, boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' },
    tooltipItemStyle: { color: tooltipTextColor },
    tooltipLabelStyle: { color: tooltipTextColor },
  };
};

/** Imperative snapshot of the current chart theme. Non-reactive — only use
 * outside React render (event handlers, canvas drawing at export time). */
export const getChartTheme = (): ChartColors => resolveChartColors(isDarkModeActive());

/** Imperative snapshot of just the tooltip style, for callers that only need it. */
export const getTooltipContentStyle = (): Record<string, string> => getChartTheme().tooltipContentStyle;
