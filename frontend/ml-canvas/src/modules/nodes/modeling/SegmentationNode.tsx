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
 * all, since neither applies to unsupervised clustering. Ships with K-Means,
 * Mini-Batch K-Means, Gaussian Mixture, and Birch — all genuinely support
 * out-of-sample `.predict()`, so they're deployable for inference (unlike
 * DBSCAN/Agglomerative/OPTICS, which only implement `fit_predict()`).
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
