import { ensureArrayOfString } from '../../sharedUtils';

const TRIM_MODES = ['both', 'leading', 'trailing'] as const;

export type TrimMode = (typeof TRIM_MODES)[number];

export const DEFAULT_TRIM_MODE: TrimMode = 'both';

const TEXTUAL_TYPE_TOKENS = ['string', 'object', 'text', 'category', 'mixed', 'varchar', 'char'];

const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 36;
const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<TrimMode, { label: string; guidance: string; example: string }> = {
  both: {
    label: 'Trim both sides',
    guidance: 'Removes whitespace on the left and right side of values. Safe default for messy strings.',
    example: "' ACME ' → 'ACME'",
  },
  leading: {
    label: 'Trim leading whitespace',
    guidance: 'Strips whitespace at the beginning of values while preserving trailing spacing.',
    example: "' 123 Main' → '123 Main'",
  },
  trailing: {
    label: 'Trim trailing whitespace',
    guidance: 'Removes whitespace at the end of values while keeping leading spacing intact.',
    example: "'Suite B  ' → 'Suite B'",
  },
};

export type TrimModeDetails = {
  mode: TrimMode;
  label: string;
  guidance: string;
  example: string;
};

export type TrimSample = {
  value: string;
  hasIssue: boolean;
};

export type TrimSampleMap = Record<string, TrimSample[]>;

export type TrimSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type TrimColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonTextSelected: string[];
  textColumnCount: number;
  whitespaceCandidateCount: number;
  whitespaceCandidates: string[];
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

const normalizeSampleDisplay = (raw: string): string => {
  const sanitized = raw.replace(/\r/g, '\\r').replace(/\n/g, '\\n').replace(/\t/g, '\\t');
  return `'${truncateSampleValue(sanitized)}'`;
};

const detectWhitespaceIssue = (raw: string): boolean => {
  if (!raw.length) {
    return false;
  }
  if (raw.trim() !== raw) {
    return true;
  }
  if (raw.includes('  ')) {
    return true;
  }
  if (raw.includes('\n') || raw.includes('\t') || raw.includes('\r')) {
    return true;
  }
  return false;
};

export const resolveTrimMode = (primary?: unknown, fallback?: unknown): TrimMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && (TRIM_MODES as readonly string[]).includes(normalizedPrimary)) {
    return normalizedPrimary as TrimMode;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && (TRIM_MODES as readonly string[]).includes(normalizedFallback)) {
    return normalizedFallback as TrimMode;
  }
  return DEFAULT_TRIM_MODE;
};

export const getTrimModeDetails = (mode: TrimMode): TrimModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_TRIM_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

const analyzeWhitespaceCandidates = (sampleMap: TrimSampleMap): string[] => {
  const candidates: string[] = [];
  Object.entries(sampleMap).forEach(([column, samples]) => {
    if (!samples.length) {
      return;
    }
    const issueCount = samples.reduce((count, sample) => (sample.hasIssue ? count + 1 : count), 0);
    if (!issueCount) {
      return;
    }
    const share = issueCount / samples.length;
    if (issueCount >= 2 || share >= 0.5) {
      candidates.push(column);
    }
  });
  candidates.sort((a, b) => a.localeCompare(b));
  return candidates;
};

type BuildTrimColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: TrimSampleMap;
};

export const buildTrimColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
}: BuildTrimColumnSummaryInput): TrimColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  uniqueSelected.sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const textColumns = availableColumns.filter((column) => isTextLikeType(columnTypeMap[column] ?? null));
  const whitespaceCandidates = analyzeWhitespaceCandidates(sampleMap);

  const recommendedSet = new Set<string>();
  textColumns.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });
  whitespaceCandidates.forEach((column) => {
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
    whitespaceCandidateCount: whitespaceCandidates.length,
    whitespaceCandidates,
  };
};

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

export const summarizeTrimRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeNonTextSelections = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeWhitespaceCandidates = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const buildTrimSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): TrimSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: TrimSampleMap = {};
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
      const hasIssue = detectWhitespaceIssue(text);
      const display = normalizeSampleDisplay(text);
      const current = map[column] ?? [];
      if (current.find((entry) => entry.value === display)) {
        return;
      }
      if (current.length >= limit) {
        return;
      }
      map[column] = [...current, { value: display, hasIssue }];
    });
  });
  return map;
};

export const summarizeTrimSamples = (
  sampleMap: TrimSampleMap,
  columns: string[],
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): TrimSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: TrimSamplePreview[] = [];
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

export const EMPTY_TRIM_COLUMN_SUMMARY: TrimColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonTextSelected: [],
  textColumnCount: 0,
  whitespaceCandidateCount: 0,
  whitespaceCandidates: [],
};
