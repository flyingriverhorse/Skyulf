/**
 * Chart color palette — works well on both light and dark backgrounds.
 * These mid-saturation colors have enough contrast against white and gray-900.
 */
export const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00C49F', '#FFBB28', '#FF8042', '#a4de6c', '#d0ed57'];

/** Detect current dark mode state */
export const isDarkMode = (): boolean =>
    document.documentElement.classList.contains('dark');

/** Theme-aware chart axis / grid helpers */
export const getChartTheme = () => {
    const dark = isDarkMode();
    return {
        axisColor: dark ? '#9ca3af' : '#6b7280',       // gray-400 / gray-500
        gridColor: dark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
        textColor: dark ? '#e5e7eb' : '#374151',        // gray-200 / gray-700
        bgColor: dark ? '#1f2937' : '#ffffff',           // gray-800 / white
        subTextColor: dark ? '#9ca3af' : '#4b5563',      // gray-400 / gray-600
        cellTextLight: dark ? '#ffffff' : '#000000',
        cellTextDark: dark ? '#ffffff' : '#000000',
    };
};
