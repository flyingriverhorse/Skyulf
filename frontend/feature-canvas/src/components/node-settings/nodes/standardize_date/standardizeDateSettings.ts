import { ensureArrayOfString } from '../../sharedUtils';

const DATE_MODES = ['iso_date', 'iso_datetime', 'month_day_year', 'day_month_year'] as const;

export type DateMode = (typeof DATE_MODES)[number];

export const DEFAULT_DATE_MODE: DateMode = 'iso_date';

const DATE_TYPE_TOKENS = ['date', 'time', 'timestamp', 'datetime'];

const PREVIEW_COLUMN_LIMIT = 6;
const PREVIEW_WARNING_LIMIT = 4;
const PREVIEW_SAMPLE_LIMIT = 3;
const PREVIEW_SAMPLE_LENGTH = 36;
const PREVIEW_COLUMN_SAMPLE_LIMIT = 5;

const MODE_METADATA: Record<DateMode, { label: string; guidance: string; example: string }> = {
  iso_date: {
    label: 'ISO date',
    guidance: 'Formats values as YYYY-MM-DD. Ideal for analytics layers expecting canonical dates.',
    example: '2024-03-19',
  },
  iso_datetime: {
    label: 'ISO datetime',
    guidance: 'Preserves date and time with second precision (YYYY-MM-DD HH:MM:SS).',
    example: '2024-03-19 14:05:42',
  },
  month_day_year: {
    label: 'MM/DD/YYYY',
    guidance: 'Aligns with US-style date formats. Use when downstream tooling expects month/day ordering.',
    example: '03/19/2024',
  },
  day_month_year: {
    label: 'DD/MM/YYYY',
    guidance: 'Suitable for EU-style day-first formats.',
    example: '19/03/2024',
  },
};

export type DateModeOption = {
  value: DateMode;
  label: string;
  guidance: string;
  example: string;
};

export const DATE_MODE_OPTIONS: DateModeOption[] = DATE_MODES.map((mode) => ({
  value: mode,
  label: MODE_METADATA[mode].label,
  guidance: MODE_METADATA[mode].guidance,
  example: MODE_METADATA[mode].example,
}));

export type DateModeDetails = {
  mode: DateMode;
  label: string;
  guidance: string;
  example: string;
};

export type DateSampleMap = Record<string, string[]>;

export type DateSamplePreview = {
  column: string;
  values: string[];
  overflow: number;
};

export type DateColumnSummary = {
  selectedColumns: string[];
  autoDetectionActive: boolean;
  recommendedColumns: string[];
  nonDateSelected: string[];
  dateColumnCount: number;
  sampleCandidateCount: number;
  sampleCandidates: string[];
};

export type DateFormatStrategyConfig = {
  mode: DateMode;
  columns: string[];
  autoDetect: boolean;
};

export type DateColumnOption = {
  name: string;
  dtype?: string | null;
  samples: string[];
  isRecommended: boolean;
  isSampleCandidate: boolean;
};

type DateStrategyFallback = {
  columns: string[];
  mode: DateMode;
  autoDetect: boolean;
};

const DATE_SAMPLE_PATTERNS = [
  /^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}/,
  /^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}/,
  /^[A-Za-z]{3,9}\s+\d{1,2},\s*\d{2,4}/,
  /^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}/,
  /^\d{4}\d{2}\d{2}$/,
];

const truncateSampleValue = (value: string): string => {
  if (value.length <= PREVIEW_SAMPLE_LENGTH) {
    return value;
  }
  return `${value.slice(0, PREVIEW_SAMPLE_LENGTH - 1)}...`;
};

const isDateLikeType = (dtype?: string | null): boolean => {
  if (!dtype) {
    return false;
  }
  const normalized = dtype.toLowerCase();
  return DATE_TYPE_TOKENS.some((token) => normalized.includes(token));
};

const matchesDatePattern = (value: string): boolean => DATE_SAMPLE_PATTERNS.some((pattern) => pattern.test(value));

export const resolveDateMode = (primary?: unknown, fallback?: unknown): DateMode => {
  const normalizedPrimary = typeof primary === 'string' ? primary.trim().toLowerCase() : '';
  if (normalizedPrimary && (DATE_MODES as readonly string[]).includes(normalizedPrimary)) {
    return normalizedPrimary as DateMode;
  }
  const normalizedFallback = typeof fallback === 'string' ? fallback.trim().toLowerCase() : '';
  if (normalizedFallback && (DATE_MODES as readonly string[]).includes(normalizedFallback)) {
    return normalizedFallback as DateMode;
  }
  return DEFAULT_DATE_MODE;
};

export const getDateModeDetails = (mode: DateMode): DateModeDetails => {
  const metadata = MODE_METADATA[mode] ?? MODE_METADATA[DEFAULT_DATE_MODE];
  return {
    mode,
    label: metadata.label,
    guidance: metadata.guidance,
    example: metadata.example,
  };
};

const analyzeSampleCandidates = (sampleMap: DateSampleMap): string[] => {
  const candidates: string[] = [];
  Object.entries(sampleMap).forEach(([column, values]) => {
    if (!values || !values.length) {
      return;
    }
    const total = values.length;
    let matches = 0;
    values.forEach((value) => {
      if (matchesDatePattern(value)) {
        matches += 1;
      }
    });
    if (!total) {
      return;
    }
    const share = matches / total;
    if (share >= 0.6 && matches >= 2) {
      candidates.push(column);
    }
  });
  candidates.sort((a, b) => a.localeCompare(b));
  return candidates;
};

type BuildDateColumnSummaryInput = {
  selectedColumns: unknown;
  availableColumns: string[];
  columnTypeMap: Record<string, string>;
  sampleMap: DateSampleMap;
};

export const buildDateColumnSummary = ({
  selectedColumns,
  availableColumns,
  columnTypeMap,
  sampleMap,
}: BuildDateColumnSummaryInput): DateColumnSummary => {
  const uniqueSelected = Array.from(new Set(ensureArrayOfString(selectedColumns)));
  uniqueSelected.sort((a, b) => a.localeCompare(b));
  const selectedSet = new Set(uniqueSelected);

  const typeMatchedDates = availableColumns.filter((column) => isDateLikeType(columnTypeMap[column] ?? null));
  const sampleCandidates = analyzeSampleCandidates(sampleMap);

  const recommendedSet = new Set<string>();
  typeMatchedDates.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });
  sampleCandidates.forEach((column) => {
    if (!selectedSet.has(column)) {
      recommendedSet.add(column);
    }
  });

  const recommendedColumns = Array.from(recommendedSet);
  recommendedColumns.sort((a, b) => a.localeCompare(b));

  const nonDateSelected = uniqueSelected.filter((column) => {
    if (isDateLikeType(columnTypeMap[column] ?? null)) {
      return false;
    }
    if (sampleCandidates.includes(column)) {
      return false;
    }
    return true;
  });

  const sortedSampleCandidates = sampleCandidates.slice().sort((a, b) => a.localeCompare(b));

  return {
    selectedColumns: uniqueSelected,
    autoDetectionActive: uniqueSelected.length === 0,
    recommendedColumns,
    nonDateSelected,
    dateColumnCount: typeMatchedDates.length,
    sampleCandidateCount: sampleCandidates.length,
    sampleCandidates: sortedSampleCandidates,
  };
};

const normalizeDateMode = (value: unknown, fallback: DateMode): DateMode => {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (normalized && (DATE_MODES as readonly string[]).includes(normalized)) {
    return normalized as DateMode;
  }
  return fallback;
};

const normalizeAutoDetect = (value: unknown): boolean => {
  if (typeof value === 'boolean') {
    return value;
  }
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') {
      return true;
    }
    if (normalized === 'false' || normalized === '0' || normalized === 'no') {
      return false;
    }
  }
  if (typeof value === 'number') {
    return value !== 0;
  }
  return false;
};

export const normalizeDateFormatStrategies = (
  value: any,
  fallback: DateStrategyFallback,
): DateFormatStrategyConfig[] => {
  if (!Array.isArray(value)) {
    if (fallback.columns.length || fallback.autoDetect) {
      return [
        {
          mode: fallback.mode,
          columns: [...fallback.columns],
          autoDetect: fallback.autoDetect,
        },
      ];
    }
    return [];
  }

  const seen = new Set<string>();
  const strategies: DateFormatStrategyConfig[] = value
    .map((entry) => {
      if (!entry || typeof entry !== 'object') {
        return null;
      }

      const rawColumns = ensureArrayOfString((entry as Record<string, unknown>).columns);
      const columns = Array.from(new Set(rawColumns.map((column) => column.trim()).filter(Boolean))).sort((a, b) =>
        a.localeCompare(b),
      );

      const mode = normalizeDateMode((entry as Record<string, unknown>).mode, fallback.mode);
      const autoDetect = normalizeAutoDetect(
        (entry as Record<string, unknown>).auto_detect ??
          (entry as Record<string, unknown>).autoDetect ??
          (entry as Record<string, unknown>).auto,
      );

      const signature = `${mode}|${columns.join(',')}|${autoDetect ? 'auto' : 'manual'}`;
      if (seen.has(signature)) {
        return null;
      }
      seen.add(signature);

      return {
        mode,
        columns,
        autoDetect,
      } as DateFormatStrategyConfig;
    })
    .filter((strategy): strategy is DateFormatStrategyConfig => Boolean(strategy));

  if (strategies.length) {
    return strategies;
  }

  if (fallback.columns.length || fallback.autoDetect) {
    return [
      {
        mode: fallback.mode,
        columns: [...fallback.columns],
        autoDetect: fallback.autoDetect,
      },
    ];
  }

  return [];
};

export const serializeDateFormatStrategies = (strategies: DateFormatStrategyConfig[]): any[] =>
  strategies.map((strategy) => {
    const payload: Record<string, any> = {
      mode: strategy.mode,
      columns: [...strategy.columns],
    };
    if (strategy.autoDetect) {
      payload.auto_detect = true;
    }
    return payload;
  });

export const formatDateModeLabel = (mode: DateMode): string => MODE_METADATA[mode]?.label ?? MODE_METADATA.iso_date.label;

const formatPreviewList = (values: string[], limit: number): string[] => values.slice(0, limit);

export const summarizeDateRecommendations = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeNonDateSelections = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_WARNING_LIMIT);
  return {
    preview,
    remaining: Math.max(columns.length - preview.length, 0),
  };
};

export const summarizeCandidateColumns = (columns: string[]): { preview: string[]; remaining: number } => {
  const preview = formatPreviewList(columns, PREVIEW_COLUMN_LIMIT);
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
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  return truncateSampleValue(text);
};

export const buildDateSampleMap = (
  rows: Array<Record<string, unknown>>,
  limit = PREVIEW_SAMPLE_LIMIT,
): DateSampleMap => {
  if (!Array.isArray(rows) || !rows.length) {
    return {};
  }
  const map: DateSampleMap = {};
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

export const summarizeDateSamples = (
  sampleMap: DateSampleMap,
  columns: string[],
  limit = PREVIEW_COLUMN_SAMPLE_LIMIT,
): DateSamplePreview[] => {
  if (!columns.length) {
    return [];
  }
  const previews: DateSamplePreview[] = [];
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

export const EMPTY_DATE_COLUMN_SUMMARY: DateColumnSummary = {
  selectedColumns: [],
  autoDetectionActive: true,
  recommendedColumns: [],
  nonDateSelected: [],
  dateColumnCount: 0,
  sampleCandidateCount: 0,
  sampleCandidates: [],
};
