import { ensureArrayOfString } from '../../sharedUtils';

const INVALID_VALUE_MODES = [
  'negative_to_nan',
  'zero_to_nan',
  'percentage_bounds',
  'age_bounds',
  'custom_range',
] as const;

export type InvalidValueMode = (typeof INVALID_VALUE_MODES)[number];

export const DEFAULT_INVALID_VALUE_MODE: InvalidValueMode = 'negative_to_nan';

const NUMERIC_TYPE_TOKENS = [
  'int',
  'float',
  'double',
  'numeric',
  'number',
  'decimal',
  'real',
  'long',
];

const SAMPLE_LIMIT = 3;
const PREVIEW_COLUMN_LIMIT = 6;
const WARNING_PREVIEW_LIMIT = 4;
const SAMPLE_PREVIEW_LIMIT = 5;

const MODE_METADATA: Record<InvalidValueMode, { label: string; guidance: string; example: string }> = {
  negative_to_nan: {
    label: 'Negative to missing',
    guidance: 'Sets negative readings to missing. Useful for metrics that cannot drop below zero.',
    example: 'Inventory change −3 → missing',
  },
  zero_to_nan: {
    label: 'Zero to missing',
    guidance: 'Treats placeholder zeros as missing values.',
    example: 'Temperature 0 → missing',
  },
  percentage_bounds: {
    label: 'Percentage bounds',
    guidance: 'Keeps readings within 0-100% unless you provide custom bounds.',
    example: 'Completion 135% → missing',
  },
  age_bounds: {
    label: 'Age bounds',
    guidance: 'Flags implausible ages outside a typical 0-120 range unless overridden.',
    example: 'Customer age 212 → missing',
  },
  custom_range: {
    label: 'Custom bounds',
    guidance: 'Applies your minimum/maximum limits to selected columns.',
    example: 'Metric < 10 or > 75 → missing',
  },
};

export type InvalidValueModeDetails = {
  mode: InvalidValueMode;
  label: string;
  guidance: string;
  example: string;
};

export type InvalidValueSample = {
  originalDisplay: string;
  numericValue: number;
  isNegative: boolean;
  isZero: boolean;
};

export type InvalidValueSampleMap = Record<string, InvalidValueSample[]>;

export type InvalidValueSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type InvalidValueColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonNumericSelected: string[];
  numericColumnCount: number;
  negativeCandidates: string[];
  zeroCandidates: string[];
  percentageOutliers: string[];
  ageOutliers: string[];
};

const normalizeDisplay = (value: string): string => {
  const sanitized = value.replace(/\r/g, '\\r').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
  return sanitized.length <= 60 ? sanitized : `${sanitized.slice(0, 59)}…`;
};

const coerceNumber = (value: unknown): number | null => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'boolean') {
    return value ? 1 : 0;
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    const numeric = Number(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
};

const isNumericType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return NUMERIC_TYPE_TOKENS.some((token) => normalized.includes(token));
};

export const resolveInvalidValueMode = (primary?: unknown, fallback?: unknown): InvalidValueMode => {
  const first = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (first && (INVALID_VALUE_MODES as readonly string[]).includes(first)) {
    return first as InvalidValueMode;
  }
  const second = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (second && (INVALID_VALUE_MODES as readonly string[]).includes(second)) {
    return second as InvalidValueMode;
  }
  return DEFAULT_INVALID_VALUE_MODE;
};

export const getInvalidValueModeDetails = (mode: InvalidValueMode): InvalidValueModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_INVALID_VALUE_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

const analyzeSampleMap = (sampleMap: InvalidValueSampleMap) => {
  const negative = new Set<string>();
  const zeros = new Set<string>();
  const percentage = new Set<string>();
  const age = new Set<string>();

  Object.entries(sampleMap).forEach(([column, samples]) => {
    samples.forEach((sample) => {
      if (sample.isNegative) {
        negative.add(column);
      }
      if (sample.isZero) {
        zeros.add(column);
      }
      if (sample.numericValue < 0 || sample.numericValue > 100) {
        percentage.add(column);
      }
      if (sample.numericValue < 0 || sample.numericValue > 120) {
        age.add(column);
      }
    });
  });

  const sort = (values: Set<string>) => Array.from(values).sort((a, b) => a.localeCompare(b));

  return {
    negative: sort(negative),
    zeros: sort(zeros),
    percentage: sort(percentage),
    age: sort(age),
  };
};

type BuildInvalidColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: InvalidValueSampleMap;
};

export const buildInvalidValueColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
}: BuildInvalidColumnSummaryInput): InvalidValueColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns))).sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const numericColumns = availableColumns.filter((column) => isNumericType(columnTypeMap[column] ?? null));
  const numericSet = new Set(numericColumns);

  const { negative, zeros, percentage, age } = analyzeSampleMap(sampleMap);

  const recommendedSet = new Set<string>();
  numericColumns.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });

  [...negative, ...zeros, ...percentage, ...age].forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });

  const nonNumericSelected = uniqueSelected.filter((column) => !numericSet.has(column));

  return {
    selectedColumns: uniqueSelected,
    autoDetectionActive: uniqueSelected.length === 0,
    recommendedColumns: Array.from(recommendedSet).sort((a, b) => a.localeCompare(b)),
    nonNumericSelected,
    numericColumnCount: numericColumns.length,
    negativeCandidates: negative,
    zeroCandidates: zeros,
    percentageOutliers: percentage,
    ageOutliers: age,
  };
};

export const summarizeInvalidRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = columns.slice(0, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeInvalidWarnings = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = columns.slice(0, WARNING_PREVIEW_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const buildInvalidValueSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = SAMPLE_LIMIT,
): InvalidValueSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: InvalidValueSampleMap = {};
  rows.forEach((row) => {
    if (!row || typeof row !== 'object') {
      return;
    }
    Object.entries(row).forEach(([rawColumn, rawValue]) => {
      const column = typeof rawColumn === 'string' ? rawColumn.trim() : '';
      if (!column) {
        return;
      }
      const numericValue = coerceNumber(rawValue);
      if (numericValue === null) {
        return;
      }
      const display = normalizeDisplay(String(rawValue));
      const sample: InvalidValueSample = {
        originalDisplay: display,
        numericValue,
        isNegative: numericValue < 0,
        isZero: Math.abs(numericValue) <= 1e-9,
      };
      const current = map[column] ?? [];
      if (current.find((entry) => entry.originalDisplay === display)) {
        return;
      }
      if (current.length >= limit) {
        return;
      }
      map[column] = [...current, sample];
    });
  });
  return map;
};

const DEFAULT_PERCENTAGE_LOWER = 0;
const DEFAULT_PERCENTAGE_UPPER = 100;
const DEFAULT_AGE_LOWER = 0;
const DEFAULT_AGE_UPPER = 120;

const evaluateSampleFlag = (
  sample: InvalidValueSample,
  mode: InvalidValueMode,
  minValue: number | null,
  maxValue: number | null,
): { flagged: boolean; reason: string | null } => {
  switch (mode) {
    case 'negative_to_nan': {
      const threshold = minValue ?? 0;
      if (sample.numericValue < threshold) {
        return { flagged: true, reason: `value < ${threshold}` };
      }
      if (maxValue !== null && sample.numericValue > maxValue) {
        return { flagged: true, reason: `value > ${maxValue}` };
      }
      return { flagged: false, reason: null };
    }
    case 'zero_to_nan':
      return {
        flagged: sample.isZero,
        reason: sample.isZero ? 'zero placeholder' : null,
      };
    case 'percentage_bounds': {
      const lower = minValue ?? DEFAULT_PERCENTAGE_LOWER;
      const upper = maxValue ?? DEFAULT_PERCENTAGE_UPPER;
      if (sample.numericValue < lower) {
        return { flagged: true, reason: `value < ${lower}` };
      }
      if (sample.numericValue > upper) {
        return { flagged: true, reason: `value > ${upper}` };
      }
      return { flagged: false, reason: null };
    }
    case 'age_bounds': {
      const lower = minValue ?? DEFAULT_AGE_LOWER;
      const upper = maxValue ?? DEFAULT_AGE_UPPER;
      if (sample.numericValue < lower) {
        return { flagged: true, reason: `value < ${lower}` };
      }
      if (sample.numericValue > upper) {
        return { flagged: true, reason: `value > ${upper}` };
      }
      return { flagged: false, reason: null };
    }
    case 'custom_range':
    default: {
      const hasLower = minValue !== null && minValue !== undefined;
      const hasUpper = maxValue !== null && maxValue !== undefined;
      if (!hasLower && !hasUpper) {
        return { flagged: false, reason: null };
      }
      if (hasLower && sample.numericValue < (minValue as number)) {
        return { flagged: true, reason: `value < ${minValue}` };
      }
      if (hasUpper && sample.numericValue > (maxValue as number)) {
        return { flagged: true, reason: `value > ${maxValue}` };
      }
      return { flagged: false, reason: null };
    }
  }
};

const formatPreviewLine = (
  sample: InvalidValueSample,
  mode: InvalidValueMode,
  minValue: number | null,
  maxValue: number | null,
): string => {
  const { flagged, reason } = evaluateSampleFlag(sample, mode, minValue, maxValue);
  if (!flagged) {
    return `${sample.originalDisplay} → ok`;
  }
  return `${sample.originalDisplay} → flagged (${reason ?? 'rule match'})`;
};

export const summarizeInvalidSamples = (
  sampleMap: InvalidValueSampleMap,
  columns: string[],
  mode: InvalidValueMode,
  minValue: number | null,
  maxValue: number | null,
  limit = SAMPLE_PREVIEW_LIMIT,
): InvalidValueSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: InvalidValueSamplePreview[] = [];
  for (const column of columns) {
    if (previews.length >= limit) {
      break;
    }
    const samples = sampleMap[column];
    if (!samples || !samples.length) {
      continue;
    }
    previews.push({
      column,
      values: samples.map((sample) => formatPreviewLine(sample, mode, minValue, maxValue)),
      overflow: 0,
    });
  }
  return previews;
};

export const EMPTY_INVALID_VALUE_COLUMN_SUMMARY: InvalidValueColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonNumericSelected: [],
  numericColumnCount: 0,
  negativeCandidates: [],
  zeroCandidates: [],
  percentageOutliers: [],
  ageOutliers: [],
};