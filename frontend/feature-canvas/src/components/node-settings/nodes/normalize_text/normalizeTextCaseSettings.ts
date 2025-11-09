import { ensureArrayOfString } from '../../sharedUtils';

const CASE_MODES = ['lower', 'upper', 'title', 'sentence'] as const;

export type CaseMode = (typeof CASE_MODES)[number];

export const DEFAULT_CASE_MODE: CaseMode = 'lower';

const TEXTUAL_TYPE_TOKENS = ['string', 'object', 'text', 'category', 'mixed', 'varchar', 'char'];

const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 60;
const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<CaseMode, { label: string; guidance: string; example: string }> = {
  lower: {
    label: 'Lowercase',
    guidance: 'Converts values to lowercase for consistent comparisons.',
    example: 'Acme Corp → acme corp',
  },
  upper: {
    label: 'Uppercase',
    guidance: 'Transforms text to uppercase, useful for codes or standard identifiers.',
    example: 'Acme Corp → ACME CORP',
  },
  title: {
    label: 'Title case',
    guidance: 'Capitalizes the first letter of each word, ideal for names or titles.',
    example: 'ACME CORP → Acme Corp',
  },
  sentence: {
    label: 'Sentence case',
    guidance: 'Capitalizes the first letter of each sentence while lowercasing the rest.',
    example: 'ACME CORP. DELIVERS. → Acme corp. delivers.',
  },
};

export type CaseModeDetails = {
  mode: CaseMode;
  label: string;
  guidance: string;
  example: string;
};

export type CaseSample = {
  original: string;
  lower: string;
  upper: string;
  title: string;
  sentence: string;
};

export type CaseSampleMap = Record<string, CaseSample[]>;

export type CaseSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type CaseColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonTextSelected: string[];
  textColumnCount: number;
  inconsistentColumns: string[];
};

const truncateSampleValue = (value: string): string => {
  if (value.length <= PREVIEW_SAMPLE_LENGTH) {
    return value;
  }
  return `${value.slice(0, PREVIEW_SAMPLE_LENGTH - 1)}...`;
};

const isTextLikeType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return TEXTUAL_TYPE_TOKENS.some((token) => normalized.includes(token));
};

const normalizeDisplay = (value: string): string => {
  const sanitized = value.replace(/\r/g, '\\r').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
  return `'${truncateSampleValue(sanitized)}'`;
};

const toTitleCase = (value: string): string =>
  value
    .toLowerCase()
    .split(/([\s\-_/]+)/)
    .map((segment) => {
      if (!segment.length || /[\s\-_/]+/.test(segment)) {
        return segment;
      }
      return segment.charAt(0).toUpperCase() + segment.slice(1);
    })
    .join('');

const toSentenceCase = (value: string): string => {
  const lower = value.toLowerCase();
  return lower.replace(/(^\s*[a-z])|([\.\!\?]\s*[a-z])/g, (match) => match.toUpperCase());
};

export const resolveCaseMode = (primary?: unknown, fallback?: unknown): CaseMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && (CASE_MODES as readonly string[]).includes(normalizedPrimary)) {
    return normalizedPrimary as CaseMode;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && (CASE_MODES as readonly string[]).includes(normalizedFallback)) {
    return normalizedFallback as CaseMode;
  }
  return DEFAULT_CASE_MODE;
};

export const getCaseModeDetails = (mode: CaseMode): CaseModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_CASE_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

type BuildCaseColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: CaseSampleMap;
  mode: CaseMode;
};

const detectInconsistentColumns = (sampleMap: CaseSampleMap, mode: CaseMode): string[] => {
  const inconsistent: string[] = [];
  Object.entries(sampleMap).forEach(([column, samples]) => {
    if (!samples.length) {
      return;
    }
    const mismatched = samples.some((sample) => {
      const expected = sample[mode];
      return sample.original !== expected;
    });
    if (mismatched) {
      inconsistent.push(column);
    }
  });
  inconsistent.sort((a, b) => a.localeCompare(b));
  return inconsistent;
};

export const buildCaseColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
  mode,
}: BuildCaseColumnSummaryInput): CaseColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  uniqueSelected.sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const textColumns = availableColumns.filter((column) => isTextLikeType(columnTypeMap[column] ?? null));
  const inconsistentColumns = detectInconsistentColumns(sampleMap, mode);

  const recommendedSet = new Set<string>();
  textColumns.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });
  inconsistentColumns.forEach((column) => {
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
    inconsistentColumns,
  };
};

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

export const summarizeCaseRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeCaseWarnings = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const buildCaseSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): CaseSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: CaseSampleMap = {};
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
      const sample: CaseSample = {
        original,
        lower: normalizeDisplay(text.toLowerCase()),
        upper: normalizeDisplay(text.toUpperCase()),
        title: normalizeDisplay(toTitleCase(text)),
        sentence: normalizeDisplay(toSentenceCase(text)),
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

export const summarizeCaseSamples = (
  sampleMap: CaseSampleMap,
  columns: string[],
  mode: CaseMode,
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): CaseSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: CaseSamplePreview[] = [];
  for (const column of columns) {
    if (previews.length >= limit) {
      break;
    }
    const samples = sampleMap[column];
    if (!samples || !samples.length) {
      continue;
    }
    const values = samples.map((sample) => `${sample.original} → ${sample[mode]}`);
    previews.push({
      column,
      values,
      overflow: 0,
    });
  }
  return previews;
};

export const EMPTY_CASE_COLUMN_SUMMARY: CaseColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonTextSelected: [],
  textColumnCount: 0,
  inconsistentColumns: [],
};
