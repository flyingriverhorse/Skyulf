import { ensureArrayOfString } from '../../sharedUtils';

const REGEX_MODES = ['preset_digits_only', 'preset_letters_only', 'preset_whitespace_collapse', 'custom'] as const;

export type RegexMode = (typeof REGEX_MODES)[number];

export const DEFAULT_REGEX_MODE: RegexMode = 'preset_digits_only';

const TEXTUAL_TYPE_TOKENS = ['string', 'object', 'text', 'category', 'mixed', 'varchar', 'char'];

const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 60;
const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<RegexMode, { label: string; guidance: string; example: string }> = {
  preset_digits_only: {
    label: 'Digits only preset',
    guidance: 'Keeps digits while stripping other characters. Useful for IDs or numeric codes.',
    example: 'Order #123A → 123',
  },
  preset_letters_only: {
    label: 'Letters only preset',
    guidance: 'Removes digits and symbols, retaining alphabetical characters.',
    example: 'SKU-45X → SKUX',
  },
  preset_whitespace_collapse: {
    label: 'Collapse whitespace preset',
    guidance: 'Replaces runs of whitespace with a single space, trimming leading/trailing spacing.',
    example: 'ACME\tCorp\n→ ACME Corp',
  },
  custom: {
    label: 'Custom regex',
    guidance: 'Applies your regex and replacement text. Supports JavaScript-style regular expressions.',
    example: 'Pattern: /\d+/g Replacement: "#"',
  },
};

export type RegexModeDetails = {
  mode: RegexMode;
  label: string;
  guidance: string;
  example: string;
};

export type RegexSample = {
  original: string;
  presetDigits: string;
  presetLetters: string;
  presetWhitespace: string;
  hasDigits: boolean;
  hasLetters: boolean;
  hasWhitespaceRuns: boolean;
};

export type RegexSampleMap = Record<string, RegexSample[]>;

export type RegexSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type RegexColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonTextSelected: string[];
  textColumnCount: number;
  digitsCandidates: string[];
  lettersCandidates: string[];
  whitespaceCandidates: string[];
};

const truncateSampleValue = (value: string): string => {
  if (value.length <= PREVIEW_SAMPLE_LENGTH) {
    return value;
  }
  return `${value.slice(0, PREVIEW_SAMPLE_LENGTH - 1)}...`;
};

const normalizeDisplay = (value: string): string => {
  const sanitized = value.replace(/\r/g, '\\r').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
  return `'${truncateSampleValue(sanitized)}'`;
};

const isTextLikeType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return TEXTUAL_TYPE_TOKENS.some((token) => normalized.includes(token));
};

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, ' ').trim();

export const resolveRegexMode = (primary?: unknown, fallback?: unknown): RegexMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && (REGEX_MODES as readonly string[]).includes(normalizedPrimary)) {
    return normalizedPrimary as RegexMode;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && (REGEX_MODES as readonly string[]).includes(normalizedFallback)) {
    return normalizedFallback as RegexMode;
  }
  return DEFAULT_REGEX_MODE;
};

export const getRegexModeDetails = (mode: RegexMode): RegexModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_REGEX_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

const analyzeRegexCandidates = (sampleMap: RegexSampleMap) => {
  const digitsCandidates = new Set<string>();
  const lettersCandidates = new Set<string>();
  const whitespaceCandidates = new Set<string>();

  Object.entries(sampleMap).forEach(([column, samples]) => {
    if (!samples.length) {
      return;
    }
    samples.forEach((sample) => {
      if (sample.hasDigits) {
        digitsCandidates.add(column);
      }
      if (sample.hasLetters) {
        lettersCandidates.add(column);
      }
      if (sample.hasWhitespaceRuns) {
        whitespaceCandidates.add(column);
      }
    });
  });

  const sort = (values: Set<string>) => Array.from(values).sort((a, b) => a.localeCompare(b));

  return {
    digits: sort(digitsCandidates),
    letters: sort(lettersCandidates),
    whitespace: sort(whitespaceCandidates),
  };
};

type BuildRegexColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: RegexSampleMap;
};

export const buildRegexColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
}: BuildRegexColumnSummaryInput): RegexColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  uniqueSelected.sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const textColumns = availableColumns.filter((column) => isTextLikeType(columnTypeMap[column] ?? null));
  const { digits, letters, whitespace } = analyzeRegexCandidates(sampleMap);

  const recommendedSet = new Set<string>();
  textColumns.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });
  [...digits, ...letters, ...whitespace].forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });

  const recommendedColumns = Array.from(recommendedSet).sort((a, b) => a.localeCompare(b));
  const nonTextSelected = uniqueSelected.filter((column) => !isTextLikeType(columnTypeMap[column] ?? null));

  return {
    selectedColumns: uniqueSelected,
    autoDetectionActive: uniqueSelected.length === 0,
    recommendedColumns,
    nonTextSelected,
    textColumnCount: textColumns.length,
    digitsCandidates: digits,
    lettersCandidates: letters,
    whitespaceCandidates: whitespace,
  };
};

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

export const summarizeRegexRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeRegexWarnings = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const buildRegexSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): RegexSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: RegexSampleMap = {};
  rows.forEach((row) => {
    if (!row || typeof row !== 'object') {
      return;
    }
    Object.entries(row).forEach(([rawColumn, rawValue]) => {
      const column = typeof rawColumn === 'string' ? rawColumn.trim() : '';
      if (!column) {
        return;
      }
      let text = '';
      if (rawValue === null || rawValue === undefined) {
        text = '';
      } else if (typeof rawValue === 'string') {
        text = rawValue;
      } else if (typeof rawValue === 'number' || typeof rawValue === 'boolean') {
        text = String(rawValue);
      } else {
        try {
          text = JSON.stringify(rawValue);
        } catch (error) {
          text = String(rawValue);
        }
      }
      if (!text.length) {
        return;
      }
      const original = normalizeDisplay(text);
      const sample: RegexSample = {
        original,
        presetDigits: normalizeDisplay(text.replace(/\D+/g, '')),
        presetLetters: normalizeDisplay(text.replace(/[^A-Za-z]+/g, '')),
        presetWhitespace: normalizeDisplay(collapseWhitespace(text)),
        hasDigits: /\d/.test(text),
        hasLetters: /[A-Za-z]/.test(text),
        hasWhitespaceRuns: /\s{2,}/.test(text) || /\t|\n|\r/.test(text),
      };
      const current = map[column] ?? [];
      if (current.find((entry) => entry.original === original)) {
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

const formatSamplePair = (sample: RegexSample, mode: RegexMode, replacement: string | null): string => {
  const replacementText = replacement ?? '';
  const format = (value: string) => (value.length ? value : "''");
  switch (mode) {
    case 'preset_letters_only':
      return `${format(sample.original)} → ${format(sample.presetLetters)}`;
    case 'preset_whitespace_collapse':
      return `${format(sample.original)} → ${format(sample.presetWhitespace)}`;
    case 'custom': {
      try {
        const regex = new RegExp(replacementText.split('|||')[0] ?? '', replacementText.split('|||')[1] ?? '');
        const result = normalizeDisplay(sample.original.replace(regex, replacementText.split('|||')[2] ?? ''));
        return `${format(sample.original)} → ${format(result)}`;
      } catch (error) {
        return `${format(sample.original)} → (invalid regex)`;
      }
    }
    case 'preset_digits_only':
    default:
      return `${format(sample.original)} → ${format(sample.presetDigits)}`;
  }
};

export const summarizeRegexSamples = (
  sampleMap: RegexSampleMap,
  columns: string[],
  mode: RegexMode,
  replacement: string | null,
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): RegexSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: RegexSamplePreview[] = [];
  for (const column of columns) {
    if (previews.length >= limit) {
      break;
    }
    const samples = sampleMap[column];
    if (!samples || !samples.length) {
      continue;
    }
    const values = samples.map((sample) => formatSamplePair(sample, mode, replacement));
    previews.push({
      column,
      values,
      overflow: 0,
    });
  }
  return previews;
};

export const EMPTY_REGEX_COLUMN_SUMMARY: RegexColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonTextSelected: [],
  textColumnCount: 0,
  digitsCandidates: [],
  lettersCandidates: [],
  whitespaceCandidates: [],
};
