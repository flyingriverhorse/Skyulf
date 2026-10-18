import { lazy, Suspense } from 'react';
import { Filter } from 'lucide-react';
import type { NodeDefinition } from '../../../core/types/nodes';
import { featureSelectionNumericIssue } from './featureSelection/numericValidation';
import type { FeatureSelectionConfig } from './featureSelection/types';

const FeatureSelectionSettings = lazy(() => import('./featureSelection/FeatureSelectionSettings'));

export const FeatureSelectionNode: NodeDefinition<FeatureSelectionConfig> = {
  type: 'feature_selection',
  label: 'Feature Selection',
  category: 'Preprocessing',
  description: 'Select the most important features.',
  icon: Filter,
  inputs: [{ id: 'in', label: 'Data', type: 'dataset' }],
  outputs: [{ id: 'out', label: 'Selected', type: 'dataset' }],
  settings: function LazyFeatureSelectionSettings(props) {
    return <Suspense fallback={<p role="status" className="p-4 text-xs text-muted-foreground">Loading settings...</p>}>
      <FeatureSelectionSettings {...props} />
    </Suspense>;
  },
  bodyPreview: (config) => {
    const method = config.method ?? 'variance_threshold';
    if ((method === 'select_k_best' || method === 'rfe') && config.k != null) return `${method} · k=${config.k}`;
    if (method === 'select_percentile' && config.percentile != null) return `${method} · ${config.percentile}%`;
    if (method === 'variance_threshold' && config.threshold != null) return `${method} · σ>${config.threshold}`;
    return method;
  },
  validate: (config) => {
    const numeric = featureSelectionNumericIssue(config);
    if (numeric) return numeric;
    if (config.method !== 'variance_threshold' && config.method !== 'correlation_threshold' && !config.target_column) {
      return { isValid: false, field: 'target_column', message: 'Target column is required for this method.' };
    }
    return { isValid: true };
  },
  getDefaultConfig: () => ({
    method: 'select_k_best',
    k: 10,
  }),
};
