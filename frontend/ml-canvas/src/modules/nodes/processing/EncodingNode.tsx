import { ValidationField } from '../../../components/shared/ValidationField';
import React from 'react';
import { NodeDefinition } from '../../../core/types/nodes';
import { Hash } from 'lucide-react';
import { RecommendationsPanel } from '../../../components/panels/RecommendationsPanel';
import { ColumnMultiSelect } from '../shared/ColumnMultiSelect';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import type { EncodingConfig } from './encoding/types';
import { useEncodingData } from './encoding/useEncodingData';
import { EncodingMethodSelection } from './encoding/EncodingMethodSelection';
import { EncodingMethodOptions } from './encoding/EncodingMethodOptions';
import { EncodingFeedback } from './encoding/EncodingFeedback';
import { fixedCategoriesValid } from './encoding/OnehotCategories';

function rareThresholdIssue(config: EncodingConfig) {
  const count = config.min_count === undefined ? 2 : config.min_count;
  const share = config.min_share === undefined ? 0 : config.min_share;
  if (!Number.isInteger(count) || count < 1)
    return { isValid: false, field: 'min_count', message: 'Minimum count must be a positive integer' };
  if (!Number.isFinite(share) || share < 0 || share > 1)
    return { isValid: false, field: 'min_share', message: 'Minimum share must be between 0 and 1' };
  return null;
}

function rareOptionsIssue(config: EncodingConfig) {
  const thresholdIssue = rareThresholdIssue(config);
  if (thresholdIssue) return thresholdIssue;
  const replacement = config.replacement === undefined ? 'Other' : config.replacement;
  if (typeof replacement !== 'string' || replacement.length === 0)
    return { isValid: false, field: 'replacement', message: 'Enter a nonempty replacement label' };
  return null;
}

function targetOptionsIssue(config: EncodingConfig) {
  const cv = config.cv === undefined ? 5 : config.cv;
  const seed = config.random_state === undefined ? 42 : config.random_state;
  if (!Number.isInteger(cv) || cv < 2)
    return { isValid: false, field: 'cv', message: 'Maximum CV folds must be an integer of at least 2' };
  if (seed !== null && (!Number.isInteger(seed) || seed < 0 || seed > 4294967295))
    return { isValid: false, field: 'random_state', message: 'Random seed must be an integer from 0 to 4294967295, or blank' };
  return null;
}

function methodOptionsIssue(config: EncodingConfig) {
  if (config.method === 'rare') return rareOptionsIssue(config);
  if (config.method === 'target') return targetOptionsIssue(config);
  return null;
}

const EncodingSettings: React.FC<{ config: EncodingConfig; onChange: (c: EncodingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const id = React.useId();
  const { datasetId, schema, isLoading, categoricalColumns, metrics, filteredRecommendations } = useEncodingData(nodeId);
  // Responsive layout: switch to a 2-column layout once the panel is wider than 400px.
  const [containerRef, isWide] = useIsWideContainer(400);

  return (
    <div ref={containerRef} className="p-4 space-y-4">
      {!datasetId && (
        <div className="p-2 bg-yellow-50 dark:bg-yellow-900/20 text-yellow-800 dark:text-yellow-400 text-xs rounded border border-yellow-200 dark:border-yellow-800">
          Connect a dataset node to see available columns.
        </div>
      )}

      {isLoading && !!datasetId && (
        <div className="text-xs text-muted-foreground animate-pulse">
          Loading schema...
        </div>
      )}

      <div className={`grid gap-4 ${isWide ? 'grid-cols-2' : 'grid-cols-1'}`}>
        <EncodingMethodSelection config={config} onChange={onChange} id={id} />
        <EncodingMethodOptions config={config} onChange={onChange} schema={schema} categoricalColumns={categoricalColumns} />
      </div>

      {/* Column Selection */}
      <ValidationField field="columns">
        <ColumnMultiSelect
          columns={categoricalColumns}
          selected={config.columns}
          onChange={(newCols) => { onChange({ ...config, columns: newCols,
            ...(config.categories === undefined ? {} : { categories: Object.fromEntries(newCols.map(column => [column, config.categories?.[column] ?? []])) }),
          }); }}
          label="Columns to Encode"
          variant="panel"
          isLoading={isLoading}
          emptyMessage="No columns found — connect an upstream dataset node."
          fillHeight={false}
        />
      </ValidationField>

      {/* Recommendations Section */}
      {filteredRecommendations.length > 0 && (
        <RecommendationsPanel recommendations={filteredRecommendations} />
      )}

      <EncodingFeedback metrics={metrics} />
    </div>
  );
};

export const EncodingNode: NodeDefinition<EncodingConfig> = {
  type: 'encoding',
  label: 'Encoding',
  category: 'Preprocessing',
  description: 'Encode categorical variables.',
  icon: Hash,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Encoded Data', type: 'dataset' }],
  settings: EncodingSettings,
  bodyPreview: (config) => {
    const cols = config.columns?.length ?? 0;
    const method = config.method ?? 'onehot';
    if (cols === 0) return method;
    return `${method} · ${cols} ${cols === 1 ? 'col' : 'cols'}`;
  },
  validate: (config) => {
    // Label and Ordinal intentionally operate on y when no columns selected
    if (config.columns.length === 0 && config.method !== 'label' && config.method !== 'ordinal')
      return { isValid: false, field: 'columns', message: 'Select at least one column' };
    if (config.method === 'woe' && !config.target_column)
      return { isValid: false, field: 'target_column', message: 'WOE encoding requires a binary target column' };
    if (!fixedCategoriesValid(config))
      return { isValid: false, field: 'categories', message: 'Enter unique categories for every selected column' };
    const issue = methodOptionsIssue(config);
    if (issue) return issue;
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'onehot',
    columns: [],
    drop_first: false,
    drop_original: true,
    handle_unknown: 'ignore',
    max_categories: 20,
    include_missing: false,
    n_features: 8,
    smooth: 'auto',
    target_type: 'auto',
    cv: 5,
    shuffle: true,
    random_state: 42,
    mode: 'frequency',
    min_count: 2,
    min_share: 0,
    replacement: 'Other',
    regularization: 0.5,
    unknown_value: -1,
    missing_code: -1,
    categories_order: '',
  }),
};
