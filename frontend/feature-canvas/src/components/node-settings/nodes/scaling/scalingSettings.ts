import type {
  ScalingColumnRecommendation,
  ScalingColumnStats,
  ScalingMethodDetail,
  ScalingMethodName,
} from '../../../../api';

export type { ScalingMethodName };

export type NormalizedScalingConfig = {
  columns: string[];
  defaultMethod: ScalingMethodName;
  columnMethods: Record<string, ScalingMethodName>;
  autoDetect: boolean;
  skippedColumns: string[];
};

export const SCALING_METHOD_ORDER: ScalingMethodName[] = ['standard', 'robust', 'minmax', 'maxabs'];

export const SCALING_METHOD_FALLBACK_LABELS: Record<ScalingMethodName, string> = {
  standard: 'Standard scaler',
  robust: 'Robust scaler',
  minmax: 'Min–Max scaler',
  maxabs: 'MaxAbs scaler',
};

export type ScalingConfidenceLevel = 'high' | 'medium' | 'low';

export const SCALING_CONFIDENCE_LABEL: Record<ScalingConfidenceLevel, string> = {
  high: 'High confidence',
  medium: 'Moderate confidence',
  low: 'Exploratory',
};

const normalizeColumnList = (value: any): string[] => {
  if (Array.isArray(value)) {
    return Array.from(
      new Set(value.map((column) => String(column ?? '').trim()).filter(Boolean)),
    ).sort((a, b) => a.localeCompare(b));
  }

  if (typeof value === 'string') {
    return Array.from(
      new Set(
        value
          .split(',')
          .map((segment) => segment.trim())
          .filter(Boolean),
      ),
    ).sort((a, b) => a.localeCompare(b));
  }

  return [];
};

export const normalizeScalingConfigValue = (value: any): NormalizedScalingConfig => {
  const source = value && typeof value === 'object' && !Array.isArray(value) ? value : {};

  const columns = normalizeColumnList(source.columns);

  const rawDefaultMethod = typeof source.default_method === 'string' ? source.default_method.trim().toLowerCase() : '';
  const defaultMethod = SCALING_METHOD_ORDER.includes(rawDefaultMethod as ScalingMethodName)
    ? (rawDefaultMethod as ScalingMethodName)
    : 'standard';

  const columnMethods: Record<string, ScalingMethodName> = {};
  const rawColumnMethods = source.column_methods;
  if (rawColumnMethods && typeof rawColumnMethods === 'object' && !Array.isArray(rawColumnMethods)) {
    Object.entries(rawColumnMethods).forEach(([key, method]) => {
      const column = String(key ?? '').trim();
      const methodKey = typeof method === 'string' ? method.trim().toLowerCase() : '';
      if (!column) {
        return;
      }
      if (SCALING_METHOD_ORDER.includes(methodKey as ScalingMethodName)) {
        columnMethods[column] = methodKey as ScalingMethodName;
      }
    });
  }

  const autoDetect = typeof source.auto_detect === 'boolean' ? source.auto_detect : true;
  const skippedColumns = normalizeColumnList(source.skipped_columns);

  return {
    columns,
    defaultMethod,
    columnMethods,
    autoDetect,
    skippedColumns,
  };
};

export const createScalingMethodLabelMap = (methods?: ScalingMethodDetail[] | null) => {
  const map: Record<ScalingMethodName, string> = { ...SCALING_METHOD_FALLBACK_LABELS };
  if (Array.isArray(methods)) {
    methods.forEach((detail) => {
      const key = typeof detail?.key === 'string' ? (detail.key.trim().toLowerCase() as ScalingMethodName) : null;
      const label = typeof detail?.label === 'string' ? detail.label.trim() : '';
      if (key && SCALING_METHOD_ORDER.includes(key) && label) {
        map[key] = label;
      }
    });
  }
  return map;
};

export const createScalingMethodDetailMap = (methods?: ScalingMethodDetail[] | null) => {
  const map = new Map<ScalingMethodName, ScalingMethodDetail>();
  if (Array.isArray(methods)) {
    methods.forEach((detail) => {
      const key = typeof detail?.key === 'string' ? (detail.key.trim().toLowerCase() as ScalingMethodName) : null;
      if (key && SCALING_METHOD_ORDER.includes(key)) {
        map.set(key, detail);
      }
    });
  }
  return map;
};

export type ScalingMethodOption = {
  value: ScalingMethodName;
  label: string;
};

export const createScalingMethodOptions = (labelMap: Record<ScalingMethodName, string>): ScalingMethodOption[] =>
  SCALING_METHOD_ORDER.map((method) => ({
    value: method,
    label: labelMap[method] ?? SCALING_METHOD_FALLBACK_LABELS[method] ?? method,
  }));

export type ScalingRecommendationRow = {
  column: string;
  dtype: string | null;
  recommendedMethod: ScalingMethodName;
  recommendedLabel: string;
  confidence: ScalingConfidenceLevel;
  confidenceLabel: string;
  reasons: string[];
  fallbackLabels: string[];
  stats: ScalingColumnStats;
  hasMissing: boolean;
  currentOverride: ScalingMethodName | null;
  currentMethod: ScalingMethodName;
  currentMethodLabel: string;
  isOverrideApplied: boolean;
  isSelected: boolean;
  isExcluded: boolean;
  isSkipped: boolean;
};

type BuildScalingRowsInput = {
  recommendations: ScalingColumnRecommendation[];
  scalingConfig: NormalizedScalingConfig;
  scalingMethodLabelMap: Record<ScalingMethodName, string>;
  scalingSelectedSet: Set<string>;
  scalingSkippedColumns: Set<string>;
  scalingExcludedColumns: Set<string>;
};

export const buildScalingRecommendationRows = ({
  recommendations,
  scalingConfig,
  scalingMethodLabelMap,
  scalingSelectedSet,
  scalingSkippedColumns,
  scalingExcludedColumns,
}: BuildScalingRowsInput): ScalingRecommendationRow[] => {
  if (!recommendations.length) {
    return [];
  }

  const rows: ScalingRecommendationRow[] = [];

  recommendations.forEach((entry) => {
    const column = String(entry?.column ?? '').trim();
    if (!column) {
      return;
    }

    const recommendedMethod = entry.recommended_method;
    const recommendedLabel =
      scalingMethodLabelMap[recommendedMethod] ??
      SCALING_METHOD_FALLBACK_LABELS[recommendedMethod] ??
      recommendedMethod;

    const fallbackLabels = Array.isArray(entry.fallback_methods)
      ? entry.fallback_methods
          .filter((method): method is ScalingMethodName => SCALING_METHOD_ORDER.includes(method as ScalingMethodName))
          .map((method) => {
            const methodKey = method as ScalingMethodName;
            return scalingMethodLabelMap[methodKey] ?? SCALING_METHOD_FALLBACK_LABELS[methodKey] ?? methodKey;
          })
      : [];

    const confidence = (entry.confidence as ScalingConfidenceLevel) ?? 'medium';
    const confidenceLabel = SCALING_CONFIDENCE_LABEL[confidence] ?? SCALING_CONFIDENCE_LABEL.medium;
    const columnMethods = scalingConfig.columnMethods;
    const currentOverride = columnMethods[column] ?? null;
    const currentMethod = currentOverride ?? scalingConfig.defaultMethod;
    const currentMethodLabel =
      scalingMethodLabelMap[currentMethod] ??
      SCALING_METHOD_FALLBACK_LABELS[currentMethod] ??
      currentMethod;

    const isExcluded = scalingExcludedColumns.has(column);
    const isSkipped = scalingSkippedColumns.has(column);
    const isSelected = scalingSelectedSet.has(column);
    const stats = entry.stats;

    rows.push({
      column,
      dtype: entry.dtype ?? null,
      recommendedMethod,
      recommendedLabel,
      confidence,
      confidenceLabel,
      reasons: Array.isArray(entry.reasons) ? entry.reasons : [],
      fallbackLabels,
      stats,
      hasMissing: Boolean(entry.has_missing),
      currentOverride,
      currentMethod,
      currentMethodLabel,
      isOverrideApplied: currentOverride !== null,
      isSelected,
      isExcluded,
      isSkipped,
    });
  });

  rows.sort((a, b) => a.column.localeCompare(b.column));
  return rows;
};

export const buildScalingOverrideSummary = (
  overrideColumns: string[],
  columnMethods: Record<string, ScalingMethodName>,
  labelMap: Record<ScalingMethodName, string>,
  overrideCount: number,
): string | null => {
  if (!overrideColumns.length) {
    return null;
  }

  const parts = overrideColumns.map((column) => {
    const method = columnMethods[column];
    const label = method ? labelMap[method] ?? SCALING_METHOD_FALLBACK_LABELS[method] ?? method : null;
  return label ? `${column} -> ${label}` : column;
  });

  if (overrideCount > overrideColumns.length) {
    parts.push('…');
  }

  return parts.join(', ');
};
