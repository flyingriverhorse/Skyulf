import type { FeatureNodeCatalogEntry } from '../../api';
import type { FeatureNodeParameter } from '../types/nodes';
import { AUTO_CONFIRMED_NODE_TYPES, DROP_MISSING_DISPLAY_DESCRIPTION, DROP_MISSING_LEGACY_DESCRIPTION } from '../constants/defaults';

export const PENDING_CONFIRMATION_FLAG = '__pending_confirmation__';

const COLUMN_ARRAY_KEY_PATTERNS = [
  /^columns?$/i,
  /_columns$/i,
  /_features$/i,
  /_list$/i,
  /_ids$/i,
  /^selected_/i,
  /^target_/i,
  /^include_/i,
  /^exclude_/i,
  /^skipped_/i,
  /_rules$/i,
  /_transformations$/i,
  /^strategies$/i,
  /^recommendations$/i,
];

const COLUMN_MAP_KEY_PATTERNS = [/methods$/i, /overrides$/i, /mapping$/i, /weights$/i, /assignments$/i];

export const isPlainObject = (value: any) => value && typeof value === 'object' && !Array.isArray(value);

export const cloneConfig = (value: any) => {
  if (value === undefined || value === null) {
    return {};
  }
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    return { ...value };
  }
};

export const isAutoConfirmedCatalogType = (catalogType?: string | null) =>
  typeof catalogType === 'string' && AUTO_CONFIRMED_NODE_TYPES.has(catalogType);

export const sanitizeDefaultConfigForNode = (catalogNode?: FeatureNodeCatalogEntry | null) => {
  const defaultConfig = catalogNode?.default_config;
  const nodeType = catalogNode?.type ?? '';
  const requiresConfirmation = !AUTO_CONFIRMED_NODE_TYPES.has(nodeType);

  const applyPendingState = (config: Record<string, any>) => {
    if (requiresConfirmation) {
      config[PENDING_CONFIRMATION_FLAG] = true;
    } else {
      delete config[PENDING_CONFIRMATION_FLAG];
    }
    return config;
  };

  if (defaultConfig === undefined || defaultConfig === null) {
    return applyPendingState({});
  }

  const cloned = cloneConfig(defaultConfig);
  if (!isPlainObject(cloned)) {
    if (cloned && typeof cloned === 'object') {
      return applyPendingState(cloned as Record<string, any>);
    }
    return applyPendingState({});
  }

  Object.entries(cloned).forEach(([key, value]) => {
    if (Array.isArray(value)) {
      const shouldReset = COLUMN_ARRAY_KEY_PATTERNS.some((pattern) => pattern.test(key));
      if (shouldReset) {
        cloned[key] = [];
      }
    } else if (isPlainObject(value)) {
      const shouldResetMap = COLUMN_MAP_KEY_PATTERNS.some((pattern) => pattern.test(key));
      if (shouldResetMap) {
        cloned[key] = {};
      }
    }
  });

  if ('auto_detect' in cloned) {
    cloned.auto_detect = false;
  }

  switch (catalogNode?.type) {
    case 'scale_numeric_features':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.column_methods = isPlainObject(cloned.column_methods) ? cloned.column_methods : {};
      cloned.skipped_columns = Array.isArray(cloned.skipped_columns) ? cloned.skipped_columns : [];
      cloned.default_method =
        typeof cloned.default_method === 'string' && cloned.default_method.trim() ? cloned.default_method : 'standard';
      break;
    case 'polynomial_features':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.auto_detect = Boolean(cloned.auto_detect);
      cloned.include_bias = Boolean(cloned.include_bias);
      cloned.interaction_only = Boolean(cloned.interaction_only);
      cloned.include_input_features = Boolean(cloned.include_input_features);
      cloned.degree = typeof cloned.degree === 'number' && Number.isFinite(cloned.degree) ? cloned.degree : 2;
      cloned.output_prefix =
        typeof cloned.output_prefix === 'string' && cloned.output_prefix.trim()
          ? cloned.output_prefix
          : 'poly';
      break;
    case 'drop_missing_columns':
      if ('missing_threshold' in cloned) {
        cloned.missing_threshold = null;
      }
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      break;
    case 'drop_missing_rows':
      if ('missing_threshold' in cloned) {
        cloned.missing_threshold = null;
      }
      if ('drop_if_any_missing' in cloned) {
        cloned.drop_if_any_missing = false;
      }
      break;
    case 'outlier_removal':
      cloned.method = 'manual';
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      cloned.manual_bounds = isPlainObject(cloned.manual_bounds) ? cloned.manual_bounds : {};
      break;
    case 'missing_value_indicator':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      break;
    case 'remove_duplicates':
      cloned.columns = Array.isArray(cloned.columns) ? cloned.columns : [];
      if (typeof cloned.keep === 'string') {
        const normalized = cloned.keep.trim().toLowerCase();
        cloned.keep = normalized === 'last' || normalized === 'none' ? normalized : 'first';
      } else {
        cloned.keep = 'first';
      }
      break;
    default:
      break;
  }

  return applyPendingState(cloned as Record<string, any>);
};

export const normalizeDescription = (description?: string, catalogType?: string) => {
  if (
    catalogType === 'drop_missing_columns' ||
    description === DROP_MISSING_LEGACY_DESCRIPTION
  ) {
    return DROP_MISSING_DISPLAY_DESCRIPTION;
  }
  return description;
};

export const normalizeParameters = (
  rawParameters?: FeatureNodeParameter[] | null
): FeatureNodeParameter[] => {
  if (!Array.isArray(rawParameters)) {
    return [];
  }
  return rawParameters.map((parameter) => ({ ...parameter }));
};
