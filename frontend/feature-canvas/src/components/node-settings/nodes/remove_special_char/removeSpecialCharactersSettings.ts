import { ensureArrayOfString } from '../../sharedUtils';

const REMOVE_SPECIAL_MODES = [
  'keep_alphanumeric',
  'keep_alphanumeric_space',
  'letters_only',
  'digits_only',
] as const;

export type RemoveSpecialMode = (typeof REMOVE_SPECIAL_MODES)[number];

export const DEFAULT_REMOVE_SPECIAL_MODE: RemoveSpecialMode = 'keep_alphanumeric';

const TEXTUAL_TYPE_TOKENS = ['string', 'object', 'text', 'category', 'mixed', 'varchar', 'char'];

const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 36;
const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<RemoveSpecialMode, { label: string; guidance: string; example: string }> = {
  keep_alphanumeric: {
    label: 'Letters & digits',
    guidance: 'Removes punctuation and symbols, leaving only letters and numbers.',
    example: 'ACME-Co. → ACMEC o',
  },
  keep_alphanumeric_space: {
    label: 'Letters, digits & spaces',
    guidance: 'Strips punctuation while preserving spaces for readability.',
    example: 'ACME-Co. → ACME Co',
  },
  letters_only: {
    label: 'Letters only',
    guidance: 'Keeps alphabetic characters and removes digits, punctuation, and symbols.',
    example: 'R2-D2 → RD',
  },
  digits_only: {
    label: 'Digits only',
    guidance: 'Preserves numbers while removing letters, punctuation, and symbols.',
    example: 'Order #123A → 123',
  },
};

export type RemoveSpecialModeDetails = {
  mode: RemoveSpecialMode;
  label: string;
  guidance: string;
  example: string;
};

export type SpecialSample = {
  value: string;
  hasSpecial: boolean;
  hasDigits: boolean;
  hasLetters: boolean;
};

export type SpecialSampleMap = Record<string, SpecialSample[]>;

export type SpecialSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type SpecialColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonTextSelected: string[];
  textColumnCount: number;
  specialCandidateCount: number;
  specialCandidates: string[];
  digitsOnlyCandidates: string[];
  lettersOnlyCandidates: string[];
};

const truncateSampleValue = (value: string): string => {
  if (value.length <= PREVIEW_SAMPLE_LENGTH) {
    return value;
  }
  return `${value.slice(0, PREVIEW_SAMPLE_LENGTH - 1)}...`;
};

const normalizeSampleDisplay = (raw: string): string => {
  const sanitized = raw.replace(/\r/g, '\\r').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
  return `'${truncateSampleValue(sanitized)}'`;
};

const isTextLikeType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return TEXTUAL_TYPE_TOKENS.some((token) => normalized.includes(token));
};

const hasSpecialCharacter = (value: string): boolean => /[^0-9A-Za-z\s]/.test(value);
const hasLetter = (value: string): boolean => /[A-Za-z]/.test(value);
const hasDigit = (value: string): boolean => /[0-9]/.test(value);

export const resolveRemoveSpecialMode = (primary?: unknown, fallback?: unknown): RemoveSpecialMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && (REMOVE_SPECIAL_MODES as readonly string[]).includes(normalizedPrimary)) {
    return normalizedPrimary as RemoveSpecialMode;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && (REMOVE_SPECIAL_MODES as readonly string[]).includes(normalizedFallback)) {
    return normalizedFallback as RemoveSpecialMode;
  }
  return DEFAULT_REMOVE_SPECIAL_MODE;
};

export const getRemoveSpecialModeDetails = (mode: RemoveSpecialMode): RemoveSpecialModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_REMOVE_SPECIAL_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

const analyzeSpecialCandidates = (sampleMap: SpecialSampleMap) => {
  const specialCandidates = new Set<string>();
  const digitsOnly = new Set<string>();
  const lettersOnly = new Set<string>();

  Object.entries(sampleMap).forEach(([column, samples]) => {
    if (!samples.length) {
      return;
    }
    let specialCount = 0;
    let digitsCount = 0;
    let lettersCount = 0;
    samples.forEach((sample) => {
      if (sample.hasSpecial) {
        specialCount += 1;
      }
      if (sample.hasDigits) {
        digitsCount += 1;
      }
      if (sample.hasLetters) {
        lettersCount += 1;
      }
    });
    if (specialCount > 0) {
      specialCandidates.add(column);
    }
    if (digitsCount > 0 && lettersCount === 0) {
      digitsOnly.add(column);
    }
    if (lettersCount > 0 && digitsCount === 0) {
      lettersOnly.add(column);
    }
  });

  const sortedSpecial = Array.from(specialCandidates).sort((a, b) => a.localeCompare(b));
  const sortedDigitsOnly = Array.from(digitsOnly).sort((a, b) => a.localeCompare(b));
  const sortedLettersOnly = Array.from(lettersOnly).sort((a, b) => a.localeCompare(b));

  return {
    specialCandidates: sortedSpecial,
    digitsOnlyCandidates: sortedDigitsOnly,
    lettersOnlyCandidates: sortedLettersOnly,
  };
};

type BuildSpecialColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: SpecialSampleMap;
};

export const buildSpecialColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
}: BuildSpecialColumnSummaryInput): SpecialColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  uniqueSelected.sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const textColumns = availableColumns.filter((column) => isTextLikeType(columnTypeMap[column] ?? null));
  const { specialCandidates, digitsOnlyCandidates, lettersOnlyCandidates } = analyzeSpecialCandidates(sampleMap);

  const recommendedSet = new Set<string>();
  textColumns.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });
  specialCandidates.forEach((column) => {
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
    specialCandidateCount: specialCandidates.length,
    specialCandidates,
    digitsOnlyCandidates,
    lettersOnlyCandidates,
  };
};

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

export const summarizeSpecialRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeSpecialWarnings = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const buildSpecialSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): SpecialSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: SpecialSampleMap = {};
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
      const normalized = normalizeSampleDisplay(text);
      const hasSpecial = hasSpecialCharacter(text);
      const sample: SpecialSample = {
        value: normalized,
        hasSpecial,
        hasDigits: hasDigit(text),
        hasLetters: hasLetter(text),
      };
      const current = map[column] ?? [];
      if (current.find((entry) => entry.value === normalized)) {
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

export const summarizeSpecialSamples = (
  sampleMap: SpecialSampleMap,
  columns: string[],
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): SpecialSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: SpecialSamplePreview[] = [];
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
      values: samples.map((entry) => entry.value),
      overflow: 0,
    });
  }
  return previews;
};

export const EMPTY_SPECIAL_COLUMN_SUMMARY: SpecialColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonTextSelected: [],
  textColumnCount: 0,
  specialCandidateCount: 0,
  specialCandidates: [],
  digitsOnlyCandidates: [],
  lettersOnlyCandidates: [],
};
