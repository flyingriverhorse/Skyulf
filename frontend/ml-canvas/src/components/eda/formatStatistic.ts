/** Missing or non-finite statistics are unknown, never a measured zero. */
export function formatStatistic(value: number | null | undefined, digits: number, suffix = ''): string {
    return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : '—';
}
