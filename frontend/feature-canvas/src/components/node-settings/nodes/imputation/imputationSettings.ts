export type ImputationStrategyMethod = 'mean' | 'median' | 'mode' | 'knn' | 'regression' | 'mice';

export type ImputationStrategyOptions = {
  neighbors?: number | null;
  max_iter?: number | null;
};

export type ImputationStrategyConfig = {
  method: ImputationStrategyMethod;
  columns: string[];
  options?: ImputationStrategyOptions;
};

export type ImputerColumnOption = {
  name: string;
  missingPercentage: number | null;
  dtype?: string | null;
  mean?: number | null;
  median?: number | null;
  mode?: string | number | null;
};

export type ImputationMethodOption = { value: ImputationStrategyMethod; label: string };

export const BASIC_IMPUTATION_METHOD_OPTIONS: ImputationMethodOption[] = [
  { value: 'mean', label: 'Mean (numeric)' },
  { value: 'median', label: 'Median (numeric)' },
  { value: 'mode', label: 'Mode (most frequent)' },
];

export const ADVANCED_IMPUTATION_METHOD_OPTIONS: ImputationMethodOption[] = [
  { value: 'knn', label: 'K-Nearest Neighbors (KNN)' },
  { value: 'regression', label: 'Iterative regression' },
  { value: 'mice', label: 'MICE (multivariate)' },
];

export const IMPUTATION_METHOD_OPTIONS: ImputationMethodOption[] = [
  ...BASIC_IMPUTATION_METHOD_OPTIONS,
  ...ADVANCED_IMPUTATION_METHOD_OPTIONS,
];

export const IMPUTER_MISSING_FILTER_PRESETS = [0, 5, 10, 20, 40, 60];

const NUMERIC_DTYPE_TOKENS = ['float', 'double', 'int', 'uint', 'long', 'short', 'byte', 'number', 'decimal'];
const BOOLEAN_DTYPE_TOKENS = ['bool', 'boolean'];
const NON_NUMERIC_DTYPE_TOKENS = ['object', 'string', 'category', 'datetime', 'date', 'time'];

export const isLikelyNumericColumn = (option: ImputerColumnOption) => {
  const rawDtype = typeof option.dtype === 'string' ? option.dtype.trim().toLowerCase() : '';
  if (typeof option.mean === 'number' || typeof option.median === 'number') {
    return true;
  }
  if (!rawDtype) {
    return true;
  }
  if (NON_NUMERIC_DTYPE_TOKENS.some((token) => rawDtype.includes(token))) {
    return false;
  }
  if (
    NUMERIC_DTYPE_TOKENS.some((token) => rawDtype.includes(token)) ||
    BOOLEAN_DTYPE_TOKENS.some((token) => rawDtype.includes(token))
  ) {
    return true;
  }
  return false;
};

export const isNumericImputationMethod = (method: ImputationStrategyMethod) => method !== 'mode';

export const buildDefaultOptionsForMethod = (
  method: ImputationStrategyMethod,
): ImputationStrategyOptions | undefined => {
  if (method === 'knn') {
    return { neighbors: 5 };
  }
  if (method === 'regression' || method === 'mice') {
    return { max_iter: 10 };
  }
  return undefined;
};

export const sanitizeOptionsForMethod = (
  method: ImputationStrategyMethod,
  options?: ImputationStrategyOptions,
): ImputationStrategyOptions | undefined => {
  if (method === 'knn') {
    const candidate = options?.neighbors;
    const numeric = Number(candidate);
    const neighbors = Number.isFinite(numeric) ? Math.max(1, Math.round(numeric)) : 5;
    return { neighbors };
  }
  if (method === 'regression' || method === 'mice') {
    const candidate = options?.max_iter;
    const numeric = Number(candidate);
    const maxIter = Number.isFinite(numeric) ? Math.max(1, Math.round(numeric)) : 10;
    return { max_iter: maxIter };
  }
  return undefined;
};

export const normalizeImputationStrategies = (
  value: any,
  allowedMethods: ImputationStrategyMethod[],
): ImputationStrategyConfig[] => {
  if (!Array.isArray(value)) {
    return [];
  }
  const methodFallback = allowedMethods[0] ?? 'mean';
  const allowedSet = new Set(allowedMethods);
  return value
    .map((entry) => {
      if (!entry || typeof entry !== 'object') {
        return null;
      }
      const rawMethod = typeof entry.method === 'string' ? entry.method.trim().toLowerCase() : methodFallback;
      const method = allowedSet.has(rawMethod as ImputationStrategyMethod)
        ? (rawMethod as ImputationStrategyMethod)
        : methodFallback;

      let columns: string[] = [];
      if (Array.isArray(entry.columns)) {
        columns = entry.columns.map((column: any) => String(column).trim()).filter(Boolean);
      } else if (typeof entry.columns === 'string') {
        columns = entry.columns
          .split(',')
          .map((column: string) => column.trim())
          .filter(Boolean);
      }

      const rawOptions =
        entry.options && typeof entry.options === 'object'
          ? (entry.options as Record<string, any>)
          : ({} as Record<string, any>);

      let options: ImputationStrategyOptions | undefined;
      if (method === 'knn') {
        const candidate = rawOptions.neighbors ?? entry.neighbors;
        const numeric = Number(candidate);
        options = {
          neighbors: Number.isFinite(numeric) ? Math.max(1, Math.round(numeric)) : 5,
        };
      } else if (method === 'regression' || method === 'mice') {
        const candidate = rawOptions.max_iter ?? rawOptions.maxIter ?? entry.max_iter ?? entry.maxIter;
        const numeric = Number(candidate);
        options = {
          max_iter: Number.isFinite(numeric) ? Math.max(1, Math.round(numeric)) : 10,
        };
      }

      const sanitizedOptions = sanitizeOptionsForMethod(method, options ?? buildDefaultOptionsForMethod(method));

      return {
        method,
        columns,
        options: sanitizedOptions,
      } as ImputationStrategyConfig;
    })
    .filter((strategy): strategy is ImputationStrategyConfig => Boolean(strategy));
};

export const serializeImputationStrategies = (strategies: ImputationStrategyConfig[]): any[] =>
  strategies.map((strategy) => {
    const payload: Record<string, any> = {
      method: strategy.method,
      columns: strategy.columns,
    };
    if (strategy.options && Object.keys(strategy.options).length) {
      payload.options = { ...strategy.options };
    }
    return payload;
  });
