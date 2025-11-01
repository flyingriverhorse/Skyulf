import type { SkewnessColumnRecommendation, SkewnessMethodDetail } from '../../../../api';

export type SkewnessTransformationMethod =
  | 'log'
  | 'square_root'
  | 'cube_root'
  | 'reciprocal'
  | 'square'
  | 'exponential'
  | 'box_cox'
  | 'yeo_johnson';

export type SkewnessTransformationConfig = {
  column: string;
  method: SkewnessTransformationMethod;
};

export type SkewnessViewMode = 'recommended' | 'all';

export const SKEWNESS_METHOD_ORDER: SkewnessTransformationMethod[] = [
  'log',
  'square_root',
  'cube_root',
  'reciprocal',
  'square',
  'exponential',
  'box_cox',
  'yeo_johnson',
];

export const DEFAULT_SKEWNESS_METHOD: SkewnessTransformationMethod = 'log';

export const isSkewnessTransformationMethod = (value: any): value is SkewnessTransformationMethod =>
  typeof value === 'string' && SKEWNESS_METHOD_ORDER.includes(value as SkewnessTransformationMethod);

export const normalizeSkewnessTransformations = (value: any): SkewnessTransformationConfig[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => {
      if (!entry || typeof entry !== 'object') {
        return null;
      }
      const rawColumn = typeof entry.column === 'string' ? entry.column.trim() : '';
      const rawMethod = typeof entry.method === 'string' ? entry.method.trim() : '';
      if (!rawColumn) {
        return null;
      }
      const method = isSkewnessTransformationMethod(rawMethod) ? rawMethod : DEFAULT_SKEWNESS_METHOD;
      return {
        column: rawColumn,
        method,
      };
    })
    .filter((item): item is SkewnessTransformationConfig => Boolean(item));
};

export const dedupeSkewnessTransformations = (
  entries: SkewnessTransformationConfig[],
): SkewnessTransformationConfig[] => {
  const seen = new Set<string>();
  const result: SkewnessTransformationConfig[] = [];
  entries.forEach((entry) => {
    const key = `${entry.column}__${entry.method}`;
    if (seen.has(key)) {
      return;
    }
    seen.add(key);
    result.push(entry);
  });
  return result;
};

export const SKEWNESS_DIRECTION_LABEL: Record<'left' | 'right', string> = {
  left: 'Left skew',
  right: 'Right skew',
};

export const FALLBACK_SKEWNESS_LABELS: Record<SkewnessTransformationMethod, string> = {
  log: 'Logarithmic',
  square_root: 'Square Root',
  cube_root: 'Cube Root',
  reciprocal: 'Reciprocal (Inverse)',
  square: 'Square',
  exponential: 'Exponential',
  box_cox: 'Box–Cox',
  yeo_johnson: 'Yeo–Johnson',
};

export const createSkewnessMethodLabelMap = (details?: SkewnessMethodDetail[] | null) => {
  const map: Record<SkewnessTransformationMethod, string> = { ...FALLBACK_SKEWNESS_LABELS };
  if (Array.isArray(details)) {
    details.forEach((detail) => {
      const rawKey = typeof detail?.key === 'string' ? detail.key.trim().toLowerCase() : '';
      const label = typeof detail?.label === 'string' ? detail.label.trim() : '';
      if (!rawKey || !label) {
        return;
      }
      if (isSkewnessTransformationMethod(rawKey)) {
        map[rawKey] = label;
      }
    });
  }
  return map;
};

export const createSkewnessMethodDetailMap = (details?: SkewnessMethodDetail[] | null) => {
  const map = new Map<SkewnessTransformationMethod, SkewnessMethodDetail>();
  if (Array.isArray(details)) {
    details.forEach((detail) => {
      const rawKey = typeof detail?.key === 'string' ? detail.key.trim().toLowerCase() : '';
      if (isSkewnessTransformationMethod(rawKey)) {
        map.set(rawKey, detail);
      }
    });
  }
  return map;
};

export type SkewnessTableRow = {
  column: string;
  recommendedMethods: SkewnessTransformationMethod[];
  primaryMethod: SkewnessTransformationMethod | null;
  directionLabel: string | null;
  magnitudeLabel: string | null;
  skewnessValue: number | null;
  summary: string | null;
  selectedMethod: SkewnessTransformationMethod | null;
};

export type SkewnessTableGroup = {
  key: string;
  label: string | null;
  method: SkewnessTransformationMethod | null;
  rows: SkewnessTableRow[];
};

export const buildSkewnessRows = ({
  viewMode,
  recommendations,
  columnOptions,
  recommendationMap,
  transformationMap,
}: {
  viewMode: SkewnessViewMode;
  recommendations: SkewnessColumnRecommendation[];
  columnOptions: string[];
  recommendationMap: Map<string, SkewnessColumnRecommendation>;
  transformationMap: Map<string, SkewnessTransformationMethod>;
}): SkewnessTableRow[] => {
  const sourceColumns =
    viewMode === 'recommended'
      ? recommendations
          .map((entry) => (entry?.column ? entry.column : null))
          .filter((column): column is string => Boolean(column))
      : columnOptions;

  const seen = new Set<string>();
  const rows: SkewnessTableRow[] = [];

  sourceColumns.forEach((rawColumn) => {
    const normalized = typeof rawColumn === 'string' ? rawColumn.trim() : '';
    if (!normalized || seen.has(normalized)) {
      return;
    }
    seen.add(normalized);

    const recommendation = recommendationMap.get(normalized) ?? null;
    const recommendedMethods = recommendation
      ? recommendation.recommended_methods.filter(isSkewnessTransformationMethod)
      : [];
    const primaryMethod = recommendedMethods.length ? recommendedMethods[0] : null;
    const directionLabel = recommendation?.direction
      ? SKEWNESS_DIRECTION_LABEL[recommendation.direction] ?? recommendation.direction
      : null;
    const magnitudeLabel = recommendation?.magnitude
      ? `${recommendation.magnitude.charAt(0).toUpperCase()}${recommendation.magnitude.slice(1)}`
      : null;
    const skewnessValue =
      recommendation && typeof recommendation.skewness === 'number' && Number.isFinite(recommendation.skewness)
        ? recommendation.skewness
        : null;
    const summary = recommendation?.summary ?? null;
    const selectedMethod = transformationMap.get(normalized) ?? null;

    rows.push({
      column: normalized,
      recommendedMethods,
      primaryMethod,
      directionLabel,
      magnitudeLabel,
      skewnessValue,
      summary,
      selectedMethod,
    });
  });

  rows.sort((a, b) => {
    const aValue = typeof a.skewnessValue === 'number' ? Math.abs(a.skewnessValue) : -Infinity;
    const bValue = typeof b.skewnessValue === 'number' ? Math.abs(b.skewnessValue) : -Infinity;
    if (aValue !== bValue) {
      return bValue - aValue;
    }
    const aHasRecommendation = Boolean(a.primaryMethod);
    const bHasRecommendation = Boolean(b.primaryMethod);
    if (aHasRecommendation !== bHasRecommendation) {
      return aHasRecommendation ? -1 : 1;
    }
    return a.column.localeCompare(b.column);
  });

  return rows;
};

export const buildSkewnessTableGroups = (
  rows: SkewnessTableRow[],
  groupByMethod: boolean,
  resolveLabel: (method: SkewnessTransformationMethod) => string,
): SkewnessTableGroup[] => {
  if (!groupByMethod) {
    return [
      {
        key: 'all',
        label: null,
        method: null,
        rows,
      },
    ];
  }

  const fallbackKey = 'no-recommendation';
  const bucket = new Map<string, SkewnessTableGroup>();

  rows.forEach((row) => {
    const method = row.primaryMethod ?? null;
    const key = method ?? fallbackKey;
    const label = method ? resolveLabel(method) : 'No recommended action';
    const entry = bucket.get(key) ?? { key, label, method, rows: [] };
    entry.rows.push(row);
    bucket.set(key, entry);
  });

  const result = Array.from(bucket.values());
  result.sort((a, b) => {
    const weight = (method: SkewnessTransformationMethod | null) =>
      method ? SKEWNESS_METHOD_ORDER.indexOf(method) : SKEWNESS_METHOD_ORDER.length;
    const diff = weight(a.method) - weight(b.method);
    if (diff !== 0) {
      return diff;
    }
    if (a.rows.length !== b.rows.length) {
      return b.rows.length - a.rows.length;
    }
    const aLabel = a.label ?? '';
    const bLabel = b.label ?? '';
    return aLabel.localeCompare(bLabel);
  });

  return result;
};
