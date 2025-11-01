import { nanoid } from 'nanoid/non-secure';
import type { FeatureMathNodeSignal } from '../../../../api';

export type FeatureMathOperationType =
  | 'arithmetic'
  | 'ratio'
  | 'stat'
  | 'similarity'
  | 'datetime_extract';

export type FeatureMathOperationDraft = {
  id: string;
  type: FeatureMathOperationType;
  method: string;
  inputColumns: string[];
  secondaryColumns: string[];
  constants: number[];
  outputColumn: string;
  outputPrefix: string;
  datetimeFeatures: string[];
  timezone: string;
  fillna: number | null;
  roundDigits: number | null;
  normalize: boolean;
  epsilon: number | null;
  allowOverwrite: boolean | null;
  description?: string;
};

export type FeatureMathOperationSummary = {
  id: string;
  label: string;
  status?: FeatureMathNodeSignal['operations'][number]['status'];
  message?: string;
  outputColumns: string[];
};

const ARITHMETIC_METHODS: Array<{ value: string; label: string }> = [
  { value: 'add', label: 'Addition (+)' },
  { value: 'subtract', label: 'Subtraction (−)' },
  { value: 'multiply', label: 'Multiplication (×)' },
  { value: 'divide', label: 'Division (÷)' },
];

const STAT_METHODS: Array<{ value: string; label: string }> = [
  { value: 'sum', label: 'Sum' },
  { value: 'mean', label: 'Mean' },
  { value: 'min', label: 'Minimum' },
  { value: 'max', label: 'Maximum' },
  { value: 'std', label: 'Standard deviation' },
  { value: 'median', label: 'Median' },
  { value: 'count', label: 'Count (non-missing)' },
  { value: 'range', label: 'Range (max - min)' },
];

const SIMILARITY_METHODS: Array<{ value: string; label: string }> = [
  { value: 'token_sort_ratio', label: 'Token sort ratio' },
  { value: 'token_set_ratio', label: 'Token set ratio' },
  { value: 'ratio', label: 'Basic ratio' },
];

export const DATETIME_FEATURE_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'year', label: 'Year' },
  { value: 'quarter', label: 'Quarter' },
  { value: 'month', label: 'Month' },
  { value: 'month_name', label: 'Month name' },
  { value: 'week', label: 'ISO week number' },
  { value: 'day', label: 'Day of month' },
  { value: 'day_name', label: 'Day name' },
  { value: 'weekday', label: 'Weekday index (Mon=0)' },
  { value: 'is_weekend', label: 'Is weekend' },
  { value: 'hour', label: 'Hour' },
  { value: 'minute', label: 'Minute' },
  { value: 'second', label: 'Second' },
  { value: 'season', label: 'Season' },
  { value: 'time_of_day', label: 'Time of day bucket' },
];

const DATETIME_FEATURE_SET = new Set(DATETIME_FEATURE_OPTIONS.map((option) => option.value));

export const FEATURE_MATH_TYPE_OPTIONS: Array<{
  value: FeatureMathOperationType;
  label: string;
  description: string;
  defaultMethod: string;
}> = [
  {
    value: 'arithmetic',
    label: 'Arithmetic',
    description: 'Combine columns and constants using addition, subtraction, multiplication, or division.',
    defaultMethod: 'add',
  },
  {
    value: 'ratio',
    label: 'Ratio',
    description: 'Divide one column or group of columns by another with safe zero handling.',
    defaultMethod: 'ratio',
  },
  {
    value: 'stat',
    label: 'Statistics',
    description: 'Compute sums, averages, min/max, or dispersion metrics across columns.',
    defaultMethod: 'mean',
  },
  {
    value: 'similarity',
    label: 'Text similarity',
    description: 'Score two text columns using rapidfuzz token-based similarity.',
    defaultMethod: 'token_sort_ratio',
  },
  {
    value: 'datetime_extract',
    label: 'Datetime extraction',
    description: 'Expand datetime columns into year, month, weekday, season and similar features.',
    defaultMethod: 'datetime_extract',
  },
];

const DEFAULT_TIMEZONE = 'UTC';

const coerceStringArray = (value: unknown): string[] => {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    return Array.from(
      new Set(
        value
          .map((entry) => (typeof entry === 'string' ? entry.trim() : String(entry ?? '').trim()))
          .filter((entry) => entry.length > 0),
      ),
    ).sort((a, b) => a.localeCompare(b));
  }
  if (typeof value === 'string') {
    return coerceStringArray(value.split(','));
  }
  return [];
};

const coerceNumberArray = (value: unknown): number[] => {
  if (!value) {
    return [];
  }
  if (Array.isArray(value)) {
    const parsed: number[] = [];
    value.forEach((entry) => {
      const numeric = Number(entry);
      if (Number.isFinite(numeric)) {
        parsed.push(numeric);
      }
    });
    return parsed;
  }
  if (typeof value === 'string') {
    return coerceNumberArray(value.split(/[;,]+/));
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? [numeric] : [];
};

const coerceNumber = (value: unknown): number | null => {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const coerceBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return fallback;
    }
    return value !== 0;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (!normalized) {
      return fallback;
    }
    if (['true', '1', 'yes', 'y', 'on'].includes(normalized)) {
      return true;
    }
    if (['false', '0', 'no', 'n', 'off'].includes(normalized)) {
      return false;
    }
  }
  return fallback;
};

const normalizeMethod = (type: FeatureMathOperationType, method: unknown): string => {
  if (typeof method !== 'string') {
    return FEATURE_MATH_TYPE_OPTIONS.find((option) => option.value === type)?.defaultMethod ?? '';
  }
  const normalized = method.trim().toLowerCase();
  if (!normalized) {
    return FEATURE_MATH_TYPE_OPTIONS.find((option) => option.value === type)?.defaultMethod ?? '';
  }
  if (type === 'arithmetic') {
    return ARITHMETIC_METHODS.find((option) => option.value === normalized)?.value ?? 'add';
  }
  if (type === 'stat') {
    return STAT_METHODS.find((option) => option.value === normalized)?.value ?? 'mean';
  }
  if (type === 'similarity') {
    return SIMILARITY_METHODS.find((option) => option.value === normalized)?.value ?? 'token_sort_ratio';
  }
  if (type === 'ratio') {
    return 'ratio';
  }
  if (type === 'datetime_extract') {
    return 'datetime_extract';
  }
  return normalized;
};

const normalizeType = (value: unknown): FeatureMathOperationType => {
  if (typeof value !== 'string') {
    return 'arithmetic';
  }
  const normalized = value.trim().toLowerCase();
  if (
    normalized === 'arithmetic' ||
    normalized === 'ratio' ||
    normalized === 'stat' ||
    normalized === 'similarity' ||
    normalized === 'datetime_extract'
  ) {
    return normalized;
  }
  return 'arithmetic';
};

const normalizeDatetimeFeatures = (value: unknown): string[] => {
  const entries = coerceStringArray(value);
  if (!entries.length) {
    return ['year', 'month', 'day'];
  }
  return entries.filter((entry) => DATETIME_FEATURE_SET.has(entry));
};

const normalizeTimezone = (value: unknown): string => {
  if (typeof value !== 'string') {
    return DEFAULT_TIMEZONE;
  }
  const normalized = value.trim();
  return normalized || DEFAULT_TIMEZONE;
};

export const normalizeFeatureMathOperations = (raw: unknown): FeatureMathOperationDraft[] => {
  if (!Array.isArray(raw)) {
    return [];
  }

  const operations: FeatureMathOperationDraft[] = [];
  raw.forEach((entry, index) => {
    if (!entry || typeof entry !== 'object') {
      return;
    }
    const payload = entry as Record<string, unknown>;
    const type = normalizeType(payload.operation_type ?? payload.type);
    const idValue = payload.operation_id ?? payload.id ?? `feature_math_${index + 1}`;
    const id = typeof idValue === 'string' && idValue.trim() ? idValue.trim() : `feature_math_${index + 1}`;
    const method = normalizeMethod(type, payload.method);
    const inputColumns = coerceStringArray(payload.input_columns ?? payload.inputs ?? payload.inputColumns);
    const secondaryColumns = coerceStringArray(payload.secondary_columns ?? payload.denominator ?? payload.secondaryColumns);
    const constants = coerceNumberArray(payload.constants);
    const outputColumn = typeof payload.output_column === 'string' ? payload.output_column.trim() : '';
    const outputPrefix = typeof payload.output_prefix === 'string' ? payload.output_prefix.trim() : '';
    const datetimeFeatures =
      type === 'datetime_extract' ? normalizeDatetimeFeatures(payload.datetime_features) : [];
    const timezone =
      type === 'datetime_extract' ? normalizeTimezone(payload.timezone) : DEFAULT_TIMEZONE;
    const fillna = coerceNumber(payload.fillna);
    const roundDigits = (() => {
      const value = coerceNumber(payload.round ?? payload.round_digits);
      return Number.isInteger(value ?? NaN) ? (value as number) : value;
    })();
    const epsilon = coerceNumber(payload.epsilon);
    const normalizeValue = type === 'similarity' ? coerceBoolean(payload.normalize, false) : false;
    const allowOverwrite = payload.allow_overwrite === undefined ? null : coerceBoolean(payload.allow_overwrite, false);
    const description = typeof payload.description === 'string' ? payload.description : undefined;

    operations.push({
      id,
      type,
      method,
      inputColumns,
      secondaryColumns,
      constants,
      outputColumn,
      outputPrefix,
      datetimeFeatures,
      timezone,
      fillna,
      roundDigits,
      normalize: normalizeValue,
      epsilon,
      allowOverwrite,
      description,
    });
  });

  return operations;
};

const sanitizeOperation = (operation: FeatureMathOperationDraft) => {
  const result: Record<string, unknown> = {
    id: operation.id,
    operation_id: operation.id,
    type: operation.type,
    operation_type: operation.type,
    method: operation.method,
    input_columns: operation.inputColumns,
    secondary_columns: operation.secondaryColumns,
    constants: operation.constants,
    output_column: operation.outputColumn,
    output_prefix: operation.outputPrefix,
  };

  if (operation.type === 'datetime_extract') {
    result.datetime_features = operation.datetimeFeatures.length
      ? operation.datetimeFeatures
      : ['year', 'month', 'day'];
    result.timezone = operation.timezone || DEFAULT_TIMEZONE;
  } else if (operation.datetimeFeatures.length) {
    result.datetime_features = operation.datetimeFeatures;
  }

  if (operation.type === 'similarity') {
    result.normalize = operation.normalize;
  }

  if (operation.fillna !== null && Number.isFinite(operation.fillna)) {
    result.fillna = operation.fillna;
  }
  if (operation.roundDigits !== null && Number.isFinite(operation.roundDigits)) {
    result.round = operation.roundDigits;
    result.round_digits = operation.roundDigits;
  }
  if (operation.epsilon !== null && Number.isFinite(operation.epsilon)) {
    result.epsilon = operation.epsilon;
  }
  if (operation.allowOverwrite !== null) {
    result.allow_overwrite = operation.allowOverwrite;
  }

  if (operation.description && operation.description.trim()) {
    result.description = operation.description.trim();
  }

  return result;
};

export const serializeFeatureMathOperations = (operations: FeatureMathOperationDraft[]): any[] =>
  operations.map((operation) => sanitizeOperation(operation));

const DEFAULT_OPERATION_INIT: Record<FeatureMathOperationType, Partial<FeatureMathOperationDraft>> = {
  arithmetic: {
    method: 'add',
    constants: [],
  },
  ratio: {
    method: 'ratio',
    constants: [],
  },
  stat: {
    method: 'mean',
    constants: [],
  },
  similarity: {
    method: 'token_sort_ratio',
    constants: [],
    normalize: false,
  },
  datetime_extract: {
    method: 'datetime_extract',
    datetimeFeatures: ['year', 'month', 'day'],
  },
};

export const createFeatureMathOperation = (
  type: FeatureMathOperationType,
  existingIds: Set<string>,
): FeatureMathOperationDraft => {
  const defaultConfig = DEFAULT_OPERATION_INIT[type] ?? {};
  let id = `math_${type}_${nanoid(6)}`;
  while (existingIds.has(id)) {
    id = `math_${type}_${nanoid(6)}`;
  }

  return {
    id,
    type,
    method: defaultConfig.method ?? FEATURE_MATH_TYPE_OPTIONS.find((option) => option.value === type)?.defaultMethod ?? 'add',
    inputColumns: [],
    secondaryColumns: [],
    constants: defaultConfig.constants ?? [],
    outputColumn: '',
    outputPrefix: '',
    datetimeFeatures:
      type === 'datetime_extract'
        ? defaultConfig.datetimeFeatures ?? ['year', 'month', 'day']
        : [],
    timezone: DEFAULT_TIMEZONE,
    fillna: null,
    roundDigits: null,
    normalize: defaultConfig.normalize ?? false,
    epsilon: null,
    allowOverwrite: null,
    description: undefined,
  };
};

export const buildFeatureMathSummaries = (
  operations: FeatureMathOperationDraft[],
  signals: FeatureMathNodeSignal[] | undefined,
): FeatureMathOperationSummary[] => {
  const signalMap = new Map<string, FeatureMathNodeSignal['operations'][number]>();
  signals?.forEach((signal) => {
    signal.operations.forEach((operation) => {
      if (!signalMap.has(operation.operation_id)) {
        signalMap.set(operation.operation_id, operation);
      }
    });
  });

  return operations.map((operation, index) => {
    const signal = signalMap.get(operation.id);
    const label = `${index + 1}. ${operation.type.replace('_', ' ')}${operation.method ? ` · ${operation.method}` : ''}`;
    return {
      id: operation.id,
      label,
      status: signal?.status,
      message: signal?.message ?? undefined,
      outputColumns: signal?.output_columns ?? [],
    };
  });
};

export const getMethodOptions = (type: FeatureMathOperationType): Array<{ value: string; label: string }> => {
  if (type === 'arithmetic') {
    return ARITHMETIC_METHODS;
  }
  if (type === 'stat') {
    return STAT_METHODS;
  }
  if (type === 'similarity') {
    return SIMILARITY_METHODS;
  }
  if (type === 'ratio') {
    return [{ value: 'ratio', label: 'Ratio (numerator / denominator)' }];
  }
  return [{ value: 'datetime_extract', label: 'Extract datetime parts' }];
};
