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
          onChange={(newCols) => { onChange({ ...config, columns: newCols }); }}
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
    regularization: 0.5,
    unknown_value: -1,
    missing_code: -1,
    categories_order: '',
  }),
};
