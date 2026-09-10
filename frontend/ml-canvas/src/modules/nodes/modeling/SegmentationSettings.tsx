import React, { useState, useId } from 'react';
import { useValidationReveal } from '../../../components/shared/ValidationField';
import { X } from 'lucide-react';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import type { ExecutionMode } from '../../../core/types/executionMode';
import { findUpstreamDatasetId } from './segmentationSettings/upstreamDataset';
import { useSegmentationModels } from './segmentationSettings/useSegmentationModels';
import { useSegmentationSubmission } from './segmentationSettings/useSegmentationSubmission';
import { SegmentationModelSection, SegmentationParametersSection } from './segmentationSettings/SegmentationSections';
import { SegmentationActionFooter } from './segmentationSettings/SegmentationActionFooter';

/** Config for the dedicated Segmentation (clustering) node.
 *
 * Deliberately its own shape — no `target_column`/CV fields — since
 * unsupervised clustering has neither.
 */
export interface SegmentationConfig {
  model_type: string;
  hyperparameters: Record<string, unknown>;
  execution_mode?: ExecutionMode;
  /** Optional column (e.g. a known label like species name) excluded from
   * training but kept around purely to help interpret which cluster
   * corresponds to which real-world group afterward (see the "Reference
   * Column" breakdown on the Segmentation results). */
  reference_column?: string | undefined;
}

/**
 * Standalone settings panel for the Segmentation node. Intentionally does
 * NOT wrap/reuse `BasicTrainingSettings` — clustering has no target column,
 * no Cross-Validation, and hyperparameters (e.g. `n_clusters`) should always
 * be visible/editable rather than hidden behind a "Customize" toggle (there
 * are no Advanced Tuning jobs to load best-params from either, since
 * Advanced Tuning excludes clustering models). Keeping this fully
 * independent means future changes to Basic Training's supervised-only UI
 * (target column, CV, tuning integration) can't accidentally affect
 * Segmentation, and vice versa.
 */
export const SegmentationSettings: React.FC<{
  config: SegmentationConfig;
  onChange: (c: SegmentationConfig) => void;
  nodeId?: string;
}> = ({ config, onChange, nodeId }) => {
  const runHelpId = useId();
  const fieldId = useId();
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_segmentation'));
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);
  const datasetId = findUpstreamDatasetId(nodeId || '', nodes, edges);
  const { data: schema } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns) : [];
  const [containerRef, isWide] = useIsWideContainer();
  const [activeTab, setActiveTab] = useState<'model' | 'params'>('model');
  useValidationReveal((field) => {
    if (field === 'model_type') setActiveTab('model');
  });
  const [showScalingAlert, setShowScalingAlert] = useState(true);
  const models = useSegmentationModels(config, onChange);
  const submission = useSegmentationSubmission(config, nodeId, datasetId, nodes, edges);

  const ModelConfigSection = <SegmentationModelSection
    config={config} onChange={onChange} fieldId={fieldId} availableColumns={availableColumns}
    availableModels={models.availableModels} isLoadingModels={models.isLoadingModels}
    requiresScaling={models.requiresScaling} changeModel={models.changeModel}
    showScalingAlert={showScalingAlert} setShowScalingAlert={setShowScalingAlert}
  />;
  const HyperparametersSection = <SegmentationParametersSection
    config={config} onChange={onChange} fieldId={fieldId}
    hyperparameters={models.hyperparameters} isLoadingDefs={models.isLoadingDefs}
  />;

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded text-xs text-blue-700 dark:text-blue-300 flex justify-between items-start gap-2">
          <span>Group rows into clusters by similarity — no target column needed.</span>
          <button
            aria-label="Dismiss segmentation information"
            onClick={() => {
              setShowInfo(false);
              sessionStorage.setItem('hide_info_segmentation', 'true');
            }}
            className="text-blue-400 hover:text-blue-600 dark:hover:text-blue-200"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {!isWide && (
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'model'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            aria-pressed={activeTab === 'model'}
            onClick={() => { setActiveTab('model'); }}
          >
            Configuration
          </button>
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'params'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            aria-pressed={activeTab === 'params'}
            onClick={() => { setActiveTab('params'); }}
          >
            Hyperparameters
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-1 pb-4 custom-scrollbar">
        {isWide ? (
          <div className="grid grid-cols-2 gap-6 h-full">
            <div className="overflow-y-auto pr-2">{ModelConfigSection}</div>
            <div className="overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-800">{HyperparametersSection}</div>
          </div>
        ) : (
          <>
            {activeTab === 'model' && ModelConfigSection}
            {activeTab === 'params' && HyperparametersSection}
          </>
        )}
      </div>

      <SegmentationActionFooter
        runHelpId={runHelpId} datasetId={datasetId} config={config}
        selectedModelItem={models.selectedModelItem} {...submission}
      />
    </div>
  );
};
