export const formatMetricValue = (value?: number | null, precision = 0): string => {
  if (value === undefined || value === null) {
    return '—';
  }
  const formatter = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: precision,
  });
  return formatter.format(value);
};

export const formatMissingPercentage = (value?: number | null): string => {
  if (value === undefined || value === null) {
    return '—';
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return '—';
  }
  const precision = numeric % 1 === 0 ? 0 : 1;
  return `${numeric.toFixed(precision)}%`;
};

export const formatNumericStat = (value?: number | null): string => {
  if (value === undefined || value === null) {
    return '—';
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return '—';
  }
  const precision = Math.abs(numeric) >= 1 ? 2 : 4;
  return formatMetricValue(numeric, precision);
};

export const formatModeStat = (value?: string | number | null): string => {
  if (value === undefined || value === null || value === '') {
    return '—';
  }
  if (typeof value === 'number') {
    return formatNumericStat(value);
  }
  return String(value);
};

export const getPriorityClass = (priority?: string | null): string | null => {
  if (!priority) {
    return null;
  }
  const normalized = priority.toLowerCase();
  if (normalized === 'critical') {
    return 'critical';
  }
  if (normalized === 'high') {
    return 'high';
  }
  if (normalized === 'medium') {
    return 'medium';
  }
  return null;
};

export const getPriorityLabel = (priority?: string | null): string | null => {
  if (!priority) {
    return null;
  }
  const normalized = priority.toLowerCase();
  if (normalized === 'critical') {
    return 'Critical';
  }
  if (normalized === 'high') {
    return 'High';
  }
  if (normalized === 'medium') {
    return 'Medium';
  }
  return priority.charAt(0).toUpperCase() + priority.slice(1);
};
