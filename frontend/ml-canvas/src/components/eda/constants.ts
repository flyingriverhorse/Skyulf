/**
 * Re-exports of the centralized chart theme (see `core/theme/chartTheme.ts`)
 * under the names EDA components already import. Kept here so existing
 * `from './constants'` imports don't need to change; new code should prefer
 * `useChartTheme()` from `core/hooks/useChartTheme` for reactive colors, or
 * import directly from `core/theme/chartTheme` for the imperative snapshot.
 */
export {
  CHART_SERIES_COLORS as COLORS,
  isDarkModeActive as isDarkMode,
  getChartTheme,
} from '../../core/theme/chartTheme';

