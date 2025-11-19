import { DATETIME_FEATURE_OPTIONS } from '../nodes/feature_math/featureMathSettings';

const FEATURE_MATH_DATETIME_KEYS = new Set(DATETIME_FEATURE_OPTIONS.map((option) => option.value));

export const sanitizeStringList = (values: string[] | undefined): string[] => {
  if (!values || !values.length) {
    return [];
  }

  return Array.from(
    new Set(values.map((value) => value.trim()).filter((value) => value.length > 0)),
  ).sort((a, b) => a.localeCompare(b));
};

export const sanitizeConstantsList = (values: number[] | undefined): number[] => {
  if (!values || !values.length) {
    return [];
  }
  return values.filter((value) => Number.isFinite(value));
};

export const sanitizeDatetimeFeaturesList = (values: string[] | undefined): string[] => {
  const sanitized = sanitizeStringList(values);
  const filtered = sanitized.filter((value) => FEATURE_MATH_DATETIME_KEYS.has(value));
  if (!filtered.length) {
    return ['year', 'month', 'day'];
  }
  return filtered;
};

export const sanitizeNumberValue = (value: number | null | undefined): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  return Number.isFinite(value) ? value : null;
};

export const sanitizeIntegerValue = (value: number | null | undefined): number | null => {
  const numeric = sanitizeNumberValue(value);
  if (numeric === null) {
    return null;
  }
  const rounded = Math.round(numeric);
  return Number.isFinite(rounded) ? rounded : null;
};

export const sanitizeTimezoneValue = (value: string | undefined): string => {
  if (typeof value !== 'string') {
    return 'UTC';
  }
  const trimmed = value.trim();
  return trimmed || 'UTC';
};
