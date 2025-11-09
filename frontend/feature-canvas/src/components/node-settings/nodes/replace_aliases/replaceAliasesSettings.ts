import { ensureArrayOfString } from '../../sharedUtils';

const ALIAS_MODES = [
  'canonicalize_country_codes',
  'normalize_boolean',
  'punctuation',
  'custom',
] as const;

export type AliasMode = (typeof ALIAS_MODES)[number];

export const DEFAULT_ALIAS_MODE: AliasMode = 'canonicalize_country_codes';

const aliasModeSet = new Set<string>(ALIAS_MODES);

const TEXTUAL_TYPE_TOKENS = ['string', 'object', 'text', 'category', 'mixed', 'varchar', 'char'];

const PREVIEW_PAIR_LIMIT = 4;
const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 36;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<AliasMode, { label: string; guidance: string; examples: string[] }> = {
  canonicalize_country_codes: {
    label: 'Country aliases',
    guidance: 'Maps abbreviations such as "UK" or "UAE" to canonical country names.',
    examples: ['UK => United Kingdom', 'UAE => United Arab Emirates'],
  },
  normalize_boolean: {
    label: 'Boolean tokens',
    guidance: 'Standardizes yes/no style values so downstream logic sees consistent tokens.',
    examples: ['y => Yes', '0 => No'],
  },
  punctuation: {
    label: 'Punctuation cleanup',
    guidance: 'Strips punctuation and squeezes whitespace to simplify fuzzy matching.',
    examples: ['C.R.M. => CRM', 'Part - Time => Part Time'],
  },
  custom: {
    label: 'Custom mappings',
    guidance: 'Applies the alias pairs you provide. Each entry should follow alias => replacement.',
    examples: ['st => Street', 'rd => Road'],
  },
};

export type AliasModeDetails = {
  mode: AliasMode;
  label: string;
  guidance: string;
  examples: string[];
};

export type AliasModeOption = {
  value: AliasMode;
  label: string;
};

export const ALIAS_MODE_OPTIONS: AliasModeOption[] = ALIAS_MODES.map((mode) => ({
  value: mode,
  label: MODE_METADATA[mode]?.label ?? mode,
}));

export type AliasStrategyConfig = {
  mode: AliasMode;
  columns: string[];
  autoDetect: boolean;
};

type AliasStrategyFallback = {
  mode: AliasMode;
  columns: string[];
  autoDetect: boolean;
};

const TRUE_TOKEN_SET = new Set(['true', '1', 'yes', 'y', 'on']);
const FALSE_TOKEN_SET = new Set(['false', '0', 'no', 'n', 'off']);

const coerceBooleanLike = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return fallback;
    }
    if (value === 1) {
      return true;
    }
    if (value === 0) {
      return false;
    }
    return fallback;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (!normalized) {
      return fallback;
    }
    if (TRUE_TOKEN_SET.has(normalized)) {
      return true;
    }
    if (FALSE_TOKEN_SET.has(normalized)) {
      return false;
    }
  }
  return fallback;
};

const coerceColumnList = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return Array.from(
      new Set(
        value
          .map((entry) => (typeof entry === 'string' ? entry.trim() : String(entry ?? '').trim()))
          .filter((entry) => entry.length > 0)
      )
    ).sort((a, b) => a.localeCompare(b));
  }
  if (typeof value === 'string') {
    return coerceColumnList(value.split(','));
  }
  return [];
};

const normalizeAliasStrategy = (
  raw: unknown,
  fallback: AliasStrategyFallback,
): AliasStrategyConfig | null => {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const entry = raw as Record<string, unknown>;
  const columns = coerceColumnList(entry.columns);
  const autoDetect = coerceBooleanLike(
    entry.auto_detect ?? entry.autoDetect ?? entry.auto,
    fallback.autoDetect,
  );
  const mode = resolveAliasMode(
    typeof entry.mode === 'string' ? entry.mode : undefined,
    fallback.mode,
  );
  if (!columns.length && !autoDetect) {
    return null;
  }
  return {
    mode,
    columns,
    autoDetect,
  };
};

export const normalizeAliasStrategies = (
  raw: unknown,
  fallback: AliasStrategyFallback,
): AliasStrategyConfig[] => {
  const sanitizedFallback: AliasStrategyConfig = {
    mode: fallback.mode,
    columns: coerceColumnList(fallback.columns),
    autoDetect: Boolean(fallback.autoDetect),
  };

  if (!raw) {
    if (!sanitizedFallback.columns.length && !sanitizedFallback.autoDetect) {
      sanitizedFallback.autoDetect = true;
    }
    return [sanitizedFallback];
  }

  if (!Array.isArray(raw)) {
    const single = normalizeAliasStrategy(raw, sanitizedFallback);
    if (single) {
      return [single];
    }
    return [sanitizedFallback];
  }

  const normalized: AliasStrategyConfig[] = [];
  raw.forEach((entry) => {
    const strategy = normalizeAliasStrategy(entry, sanitizedFallback);
    if (strategy) {
      normalized.push(strategy);
    }
  });

  if (normalized.length === 0) {
    if (!sanitizedFallback.columns.length && !sanitizedFallback.autoDetect) {
      sanitizedFallback.autoDetect = true;
    }
    return [sanitizedFallback];
  }

  return normalized;
};

export const serializeAliasStrategies = (
  strategies: AliasStrategyConfig[],
): Array<{ mode: AliasMode; columns: string[]; auto_detect: boolean }> =>
  strategies.map((strategy) => ({
    mode: strategy.mode,
    columns: [...strategy.columns],
    auto_detect: Boolean(strategy.autoDetect),
  }));

export type AliasColumnSummary = {
  selectedColumns: string[];
  nonTextSelected: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  textColumnCount: number;
  textColumns: string[];
};

export type AliasCustomPairSummary = {
  totalPairs: number;
  previewPairs: string[];
  previewOverflow: number;
  duplicates: string[];
  duplicateOverflow: number;
  invalidEntries: string[];
  invalidOverflow: number;
};

export type AliasSampleMap = Record<string, string[]>;

export type AliasSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type AliasColumnOption = {
  name: string;
  dtype: string | null | undefined;
  samples: string[];
  isRecommended: boolean;
  isTextLike: boolean;
};

const normalizeAliasKey = (value: string): string => value.replace(/[\s\W_]+/g, '').toLowerCase();

const truncateSampleValue = (value: string): string => {
  if (value.length <= PREVIEW_SAMPLE_LENGTH) {
    return value;
  }
  return `${value.slice(0, PREVIEW_SAMPLE_LENGTH - 1)}...`;
};

const isAliasMode = (value: string): value is AliasMode => aliasModeSet.has(value);

const isTextLikeType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return TEXTUAL_TYPE_TOKENS.some((token) => normalized.includes(token));
};

export const resolveAliasMode = (primary?: unknown, fallback?: unknown): AliasMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && isAliasMode(normalizedPrimary)) {
    return normalizedPrimary;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && isAliasMode(normalizedFallback)) {
    return normalizedFallback;
  }
  return DEFAULT_ALIAS_MODE;
};

export const getAliasModeDetails = (mode: AliasMode): AliasModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_ALIAS_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    examples: metadata.examples,
  };
};

type BuildAliasColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  autoDetectEnabled?: boolean;
};

export const buildAliasColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  autoDetectEnabled,
}: BuildAliasColumnSummaryInput): AliasColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  const selectedSet = new Set(uniqueSelected);
  const textColumns = availableColumns.filter((column) => isTextLikeType(columnTypeMap[column] ?? null));
  const recommendedColumns = textColumns.filter((column) => !selectedSet.has(column));
  const nonTextSelected = uniqueSelected.filter((column) => !isTextLikeType(columnTypeMap[column] ?? null));
  const autoDetectionActive =
    typeof autoDetectEnabled === 'boolean' ? autoDetectEnabled : uniqueSelected.length === 0;

  return {
    selectedColumns: uniqueSelected,
    nonTextSelected,
    autoDetectionActive,
    recommendedColumns,
    textColumnCount: textColumns.length,
    textColumns,
  };
};

const ENTRY_SPLIT_REGEX = /[\n;,]+/;

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

const splitPairText = (text: string, delimiter: '=>' | ':'): [string, string] | null => {
  const parts = text.split(delimiter);
  if (parts.length < 2) {
    return null;
  }
  const [alias, ...rest] = parts;
  const replacement = rest.join(delimiter);
  return [alias, replacement];
};

export const parseCustomAliasPairs = (raw: unknown): AliasCustomPairSummary => {
  if (!raw) {
    return {
      totalPairs: 0,
      previewPairs: [],
      previewOverflow: 0,
      duplicates: [],
      duplicateOverflow: 0,
      invalidEntries: [],
      invalidOverflow: 0,
    };
  }

  const duplicates = new Set<string>();
  const invalidEntries: string[] = [];
  const previewPairs: string[] = [];
  const seen = new Map<string, string>();
  let totalPairs = 0;

  const registerPair = (aliasRaw: unknown, replacementRaw: unknown) => {
    const alias = String(aliasRaw ?? '').trim();
    const replacement = String(replacementRaw ?? '').trim();
    if (!alias || !replacement) {
      if (alias || replacement) {
        invalidEntries.push(`${alias} => ${replacement}`.trim());
      }
      return;
    }
    const normalized = normalizeAliasKey(alias);
    if (!normalized) {
      invalidEntries.push(`${alias} => ${replacement}`);
      return;
    }
    if (seen.has(normalized)) {
      duplicates.add(alias);
    } else {
      seen.set(normalized, replacement);
      if (previewPairs.length < PREVIEW_PAIR_LIMIT) {
        previewPairs.push(`${alias} => ${replacement}`);
      }
    }
    totalPairs += 1;
  };

  if (typeof raw === 'string') {
    const entries = raw.split(ENTRY_SPLIT_REGEX);
    entries.forEach((entry) => {
      const text = entry.trim();
      if (!text) {
        return;
      }
      if (text.includes('=>')) {
        const parsed = splitPairText(text, '=>');
        if (parsed) {
          registerPair(parsed[0], parsed[1]);
        } else {
          invalidEntries.push(text);
        }
        return;
      }
      if (text.includes(':')) {
        const parsed = splitPairText(text, ':');
        if (parsed) {
          registerPair(parsed[0], parsed[1]);
        } else {
          invalidEntries.push(text);
        }
        return;
      }
      invalidEntries.push(text);
    });
    const duplicateList = Array.from(duplicates);
    const duplicatePreview = formatPreviewList(duplicateList, PREVIEW_WARNING_LIMIT);
    const invalidPreview = formatPreviewList(invalidEntries, PREVIEW_WARNING_LIMIT);
    return {
      totalPairs,
      previewPairs,
      previewOverflow: Math.max(totalPairs - previewPairs.length, 0),
      duplicates: duplicatePreview,
      duplicateOverflow: Math.max(duplicateList.length - duplicatePreview.length, 0),
      invalidEntries: invalidPreview,
      invalidOverflow: Math.max(invalidEntries.length - invalidPreview.length, 0),
    };
  }

  if (Array.isArray(raw)) {
    raw.forEach((entry) => {
      if (!entry) {
        return;
      }
      if (typeof entry === 'string') {
        const text = entry.trim();
        if (!text) {
          return;
        }
        if (text.includes('=>')) {
          const parsed = splitPairText(text, '=>');
          if (parsed) {
            registerPair(parsed[0], parsed[1]);
          } else {
            invalidEntries.push(text);
          }
        } else if (text.includes(':')) {
          const parsed = splitPairText(text, ':');
          if (parsed) {
            registerPair(parsed[0], parsed[1]);
          } else {
            invalidEntries.push(text);
          }
        } else {
          invalidEntries.push(text);
        }
        return;
      }
      if (typeof entry === 'object') {
        const { alias, replacement } = entry as { alias?: unknown; replacement?: unknown };
        registerPair(alias, replacement);
        return;
      }
      invalidEntries.push(String(entry));
    });
    const duplicateList = Array.from(duplicates);
    const duplicatePreview = formatPreviewList(duplicateList, PREVIEW_WARNING_LIMIT);
    const invalidPreview = formatPreviewList(invalidEntries, PREVIEW_WARNING_LIMIT);
    return {
      totalPairs,
      previewPairs,
      previewOverflow: Math.max(totalPairs - previewPairs.length, 0),
      duplicates: duplicatePreview,
      duplicateOverflow: Math.max(duplicateList.length - duplicatePreview.length, 0),
      invalidEntries: invalidPreview,
      invalidOverflow: Math.max(invalidEntries.length - invalidPreview.length, 0),
    };
  }

  if (typeof raw === 'object') {
    Object.entries(raw as Record<string, unknown>).forEach(([alias, replacement]) => {
      registerPair(alias, replacement);
    });
  }

  const duplicateList = Array.from(duplicates);
  const duplicatePreview = formatPreviewList(duplicateList, PREVIEW_WARNING_LIMIT);
  const invalidPreview = formatPreviewList(invalidEntries, PREVIEW_WARNING_LIMIT);
  return {
    totalPairs,
    previewPairs,
    previewOverflow: Math.max(totalPairs - previewPairs.length, 0),
    duplicates: duplicatePreview,
    duplicateOverflow: Math.max(duplicateList.length - duplicatePreview.length, 0),
    invalidEntries: invalidPreview,
    invalidOverflow: Math.max(invalidEntries.length - invalidPreview.length, 0),
  };
};

export const summarizeRecommendedColumns = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = columns.slice(0, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeNonTextColumns = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = columns.slice(0, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

const formatSampleValue = (value: unknown): string | null => {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      return null;
    }
    return truncateSampleValue(String(value));
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    return truncateSampleValue(trimmed);
  }
  try {
    const serialized = JSON.stringify(value);
    if (!serialized) {
      return null;
    }
    return truncateSampleValue(serialized);
  } catch (error) {
    return truncateSampleValue(String(value));
  }
};

export const buildAliasSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): AliasSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: AliasSampleMap = {};
  rows.forEach((row) => {
    if (!row || typeof row !== 'object') {
      return;
    }
    Object.entries(row).forEach(([rawColumn, rawValue]) => {
      const column = typeof rawColumn === 'string' ? rawColumn.trim() : '';
      if (!column) {
        return;
      }
      const formatted = formatSampleValue(rawValue);
      if (!formatted) {
        return;
      }
      const current = map[column] ?? [];
      if (current.includes(formatted)) {
        return;
      }
      if (current.length >= limit) {
        return;
      }
      map[column] = [...current, formatted];
    });
  });
  return map;
};

export const summarizeAliasSamples = (
  sampleMap: AliasSampleMap,
  columns: string[],
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): AliasSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: AliasSamplePreview[] = [];
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
      values: samples,
      overflow: 0,
    });
  }
  return previews;
};
