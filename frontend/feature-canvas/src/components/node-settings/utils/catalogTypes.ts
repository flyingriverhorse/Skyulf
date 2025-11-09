// Node catalog helpers leveraged by NodeSettingsModal to derive flag sets.
import type { Node } from 'react-flow-renderer';

export const isDatasetNode = (node: Node | null | undefined) =>
  Boolean(node?.data?.isDataset || node?.id === 'dataset-source');

export const INSPECTION_NODE_TYPES = new Set<string>([
  'binned_distribution',
  'data_preview',
  'outlier_monitor',
  'skewness_distribution',
  'dataset_profile',
  'transformer_audit',
]);

export const DATA_CONSISTENCY_TYPES = new Set<string>([
  'trim_whitespace',
  'normalize_text_case',
  'replace_aliases_typos',
  'standardize_date_formats',
  'remove_special_characters',
  'replace_invalid_values',
  'regex_replace_fix',
]);
