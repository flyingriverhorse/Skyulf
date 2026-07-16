import { useEffect, useState } from 'react';
import { isDarkModeActive, resolveChartColors, type ChartColors } from '../theme/chartTheme';

/**
 * Reactively tracks whether dark mode (the `dark` class on `<html>`) is
 * active. The app toggles this class directly (see `Layout.tsx`) without a
 * React context, so components that only read `document.documentElement`
 * once at render time go stale until something else re-renders them. This
 * hook watches the class attribute with a `MutationObserver` so charts
 * update live the moment the user flips the theme toggle.
 */
export const useIsDarkMode = (): boolean => {
  const [isDark, setIsDark] = useState(isDarkModeActive);

  useEffect(() => {
    const target = document.documentElement;
    const observer = new MutationObserver(() => { setIsDark(isDarkModeActive()); });
    observer.observe(target, { attributes: true, attributeFilter: ['class'] });
    return () => { observer.disconnect(); };
  }, []);

  return isDark;
};

/** Reactive, centralized chart color palette — see `core/theme/chartTheme.ts`
 * for the underlying values. Use this (not `getChartTheme()`) in any
 * component rendered as JSX so colors update live on theme toggle. */
export const useChartTheme = (): ChartColors => resolveChartColors(useIsDarkMode());
