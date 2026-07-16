import { Boxes } from 'lucide-react';
import { SegmentationSettings, SegmentationConfig } from './SegmentationSettings';
import { createModelingNode } from '../../../core/factories/nodeFactory';

/**
 * Dedicated Segmentation (clustering) node — fully independent of
 * `BasicTrainingSettings`/the Basic Training node. Its settings panel
 * (`SegmentationSettings`) has its own model-list fetch (clustering-tagged
 * algorithms only), its own hyperparameter handling (always visible/editable,
 * no "Customize" toggle — there's nothing to load best-params from, since
 * Advanced Tuning excludes clustering models), and no target-column/CV UI at
 * all, since neither applies to unsupervised clustering. Currently K-Means
 * only; the backend's `problem_type == "clustering"` dispatch is
 * engine-agnostic, so future algorithms with genuine out-of-sample
 * `.predict()` support can be added without further node-level changes.
 */
export const SegmentationNode = createModelingNode<SegmentationConfig>({
  type: 'SegmentationNode',
  label: 'Segmentation',
  description: 'Group rows into clusters (segments) by similarity — no target column needed.',
  icon: Boxes,
  settings: (props) => <SegmentationSettings {...props} />,
  defaultConfig: {
    model_type: 'kmeans',
    hyperparameters: {}
  },
  bodyPreview: (config) => (config.model_type ? `${config.model_type}` : null),
  validate: (config) => {
    if (!config.model_type) return { isValid: false, message: 'Select a clustering algorithm.' };
    return { isValid: true };
  },
  outputs: [{ id: 'model', label: 'Trained Model', type: 'model' }]
});
