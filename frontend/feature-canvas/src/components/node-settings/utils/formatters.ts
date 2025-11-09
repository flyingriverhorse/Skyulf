// Formatting helpers consumed by NodeSettingsModal sections when rendering preview data.
import { formatMetricValue } from '../formatting';

export const formatCellValue = (value: any): string => {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? formatMetricValue(value, Math.abs(value) >= 1 ? 2 : 4) : String(value);
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.length ? JSON.stringify(value) : '[]';
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch (error) {
      return String(value);
    }
  }
  return String(value);
};

export const formatColumnType = (value?: string | null) => {
  if (!value) {
    return 'Unknown';
  }
  return value;
};

const parseServerTimestamp = (value?: string | null): Date | null => {
  if (!value) {
    return null;
  }

  let normalized = value.trim();
  if (!normalized) {
    return null;
  }

  if (/^\d{4}-\d{2}-\d{2}\s+\d/.test(normalized)) {
    normalized = normalized.replace(' ', 'T');
  }

  if (!/(Z|z|[+-]\d{2}:?\d{2})$/.test(normalized)) {
    normalized = `${normalized}Z`;
  }

  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return parsed;
};

export const formatRelativeTime = (value?: string | null) => {
  const parsed = parseServerTimestamp(value);
  if (!parsed) {
    return null;
  }

  const diffMinutes = Math.round((Date.now() - parsed.getTime()) / 60000);
  if (diffMinutes < 1) {
    return 'just now';
  }
  if (diffMinutes === 1) {
    return '1 minute ago';
  }
  if (diffMinutes < 60) {
    return `${diffMinutes} minutes ago`;
  }
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours === 1) {
    return '1 hour ago';
  }
  if (diffHours < 24) {
    return `${diffHours} hours ago`;
  }
  const diffDays = Math.round(diffHours / 24);
  if (diffDays === 1) {
    return '1 day ago';
  }
  return `${diffDays} days ago`;
};
