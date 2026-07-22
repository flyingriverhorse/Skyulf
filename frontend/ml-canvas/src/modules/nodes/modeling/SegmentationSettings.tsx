import React, { useEffect, useState, useRef } from 'react';
import { Play, Loader2, Settings2, AlertCircle, ChevronDown, X } from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import { RegistryItem, registryApi } from '../../../core/api/registry';
import { useIsWideContainer } from '../../../core/hooks/useIsWideContainer';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';
import { warnAndBlockOnLeakage } from '../../../core/utils/pipelineLeakageValidation';
import { getIncomers } from '@xyflow/react';
import { HelpTooltip } from './components/HelpTooltip';
import { HyperparameterInput } from './components/HyperparameterInput';
import type { HyperparameterDef } from './components/types';
import type { ExecutionMode } from '../../../core/types/executionMode';
import { toast } from '../../../core/toast';

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
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_segmentation'));

  const { toggleDrawer: toggleJobDrawer, setTab, setActiveParallelRun, startPolling } = useJobStore();

  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
    const visited = new Set<string>();
    const queue = [currentNodeId];

    while (queue.length > 0) {
      const id = queue.shift();
      if (!id) continue;
      if (visited.has(id)) continue;
      visited.add(id);

      const node = nodes.find(n => n.id === id);
      if (!node) continue;

      if (id !== currentNodeId) {
        if (node.data?.datasetId) return node.data.datasetId as string;
        if (node.data?.dataset_id) return node.data.dataset_id as string;

        if (node.data?.config) {
          const nodeConfig = node.data.config as Record<string, unknown>;
          if (nodeConfig.datasetId) return nodeConfig.datasetId as string;
          if (nodeConfig.dataset_id) return nodeConfig.dataset_id as string;
        }

        if (node.data?.params) {
          const params = node.data.params as Record<string, unknown>;
          if (params.datasetId) return params.datasetId as string;
          if (params.dataset_id) return params.dataset_id as string;
        }
      }

      const incomers = getIncomers(node, nodes, edges);
      for (const incomer of incomers) {
        queue.push(incomer.id);
      }
    }
    return undefined;
  };

  const datasetId = findUpstreamDatasetId(nodeId || '');
  const { data: schema } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns) : [];

  const [containerRef, isWide] = useIsWideContainer();
  const [activeTab, setActiveTab] = useState<'model' | 'params'>('model');
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showScalingAlert, setShowScalingAlert] = useState(true);

  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');

  // Fetch clustering-tagged algorithms only.
  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingModels(true);
      try {
        const nodes = await registryApi.getAllNodes();
        const models = nodes.filter(n => {
          const isModeling = n.category === 'Model' || n.category === 'Modeling';
          const isClustering = n.tags?.includes('clustering') ?? false;
          return isModeling && isClustering;
        });
        setAvailableModels(models);
      } catch (error) {
        console.error('Failed to fetch clustering models:', error);
        setAvailableModels([
          { id: 'kmeans', name: 'K-Means', category: 'Modeling', description: '', params: {}, tags: ['clustering'] },
        ]);
      } finally {
        setIsLoadingModels(false);
      }
    };
    fetchModels();
  }, []);

  const keepCustomizationOpen = useRef(false);

  // Fetch hyperparameter definitions and always seed defaults — Segmentation
  // has no "Customize" toggle, so params must be ready to show/edit as soon
  // as the model type is known.
  useEffect(() => {
    if (config.model_type) {
      setIsLoadingDefs(true);
      jobsApi.getHyperparameters(config.model_type)
        .then((defs) => {
          const definitions = defs as HyperparameterDef[];
          setHyperparameters(definitions);

          if (keepCustomizationOpen.current || Object.keys(config.hyperparameters).length === 0) {
            const defaults: Record<string, unknown> = {};
            definitions.forEach(p => {
              defaults[p.name] = p.default;
            });
            onChange({ ...config, hyperparameters: defaults });
            keepCustomizationOpen.current = false;
          }
        })
        .catch(console.error)
        .finally(() => { setIsLoadingDefs(false); });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type]);

  const handleTrain = async () => {
    if (!nodeId) return;
    if (!datasetId) {
      toast.error('No dataset connected', 'Connect a dataset node upstream before starting training.');
      return;
    }
    try {
      const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
      if (warnAndBlockOnLeakage(pipelineConfig)) return;
      const response = await jobsApi.runPipeline({
        ...pipelineConfig,
        target_node_id: nodeId,
        job_type: 'training'
      });
      const jobCount = response.job_ids?.length || 1;
      if (jobCount > 1) {
        setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
        startPolling();
        toast.success('Parallel execution started', `${jobCount} branches submitted.`);
      } else {
        toast.success('Segmentation job submitted');
      }
      setTab('segmentation');
      toggleJobDrawer(true);
    } catch (error) {
      console.error('Failed to submit segmentation job:', error);
      toast.error('Failed to submit segmentation job', 'Check console for details.');
    }
  };

  const ModelConfigSection = (
    <div className="space-y-5 animate-in fade-in duration-300">
      <div className="space-y-4">
        <div className="space-y-1.5">
          <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</span>
          <div className="grid gap-3">
            <div>
              <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Clustering Algorithm</span>
              <div className="relative">
                <select
                  value={config.model_type}
                  onChange={(e) => {
                    if (Object.keys(config.hyperparameters).length > 0) {
                      keepCustomizationOpen.current = true;
                    }
                    onChange({ ...config, model_type: e.target.value, hyperparameters: {} });
                  }}
                  className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                  disabled={isLoadingModels}
                >
                  {availableModels.map(model => (
                    <option key={model.id} value={model.id}>{model.name}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>

              {requiresScaling && (
                <div className="mt-2 text-xs border border-blue-200 dark:border-blue-800 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 overflow-hidden transition-all">
                  <button
                    onClick={() => setShowScalingAlert(!showScalingAlert)}
                    className="w-full flex items-center justify-between p-2 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
                  >
                    <div className="flex items-center gap-2 font-semibold">
                      <AlertCircle className="w-3 h-3" />
                      <span>Scale Your Data</span>
                    </div>
                    <ChevronDown className={`w-3 h-3 transition-transform ${showScalingAlert ? 'rotate-180' : ''}`} />
                  </button>
                  {showScalingAlert && (
                    <div className="p-2 pt-0 opacity-90 animate-in slide-in-from-top-1 pl-7">
                      This model performs best with scaled features. Consider adding a &quot;Feature Scaling&quot; node.
                    </div>
                  )}
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <span className="block text-xs font-medium text-gray-700 dark:text-gray-300">Reference Column (optional)</span>
                <HelpTooltip text="A column with a known real-world label (e.g. a species/customer-type name) that you want excluded from clustering, but kept around afterward to see which cluster corresponds to which group — e.g. 'Cluster 0 is 92% setosa'. The model never sees this column." />
              </div>
              <div className="relative">
                <select
                  value={config.reference_column ?? ''}
                  onChange={(e) => onChange({ ...config, reference_column: e.target.value || undefined })}
                  className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
                  disabled={availableColumns.length === 0}
                >
                  <option value="">None</option>
                  {availableColumns.map((col) => (
                    <option key={col.name} value={col.name}>{col.name}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const HyperparametersSection = (
    <div className="space-y-4 animate-in fade-in duration-300">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <Settings2 className="w-4 h-4 text-blue-500" />
          Hyperparameters
        </h4>
      </div>

      <div className="space-y-3">
        {isLoadingDefs ? (
          <div className="flex justify-center py-4">
            <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
          </div>
        ) : (
          hyperparameters.map((param) => (
            <div key={param.name} className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
              <div className="flex justify-between items-center mb-2">
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                  {param.label}
                </label>
                {param.description && <HelpTooltip text={param.description} />}
              </div>
              {param.type === 'select' ? (
                <select
                  value={(config.hyperparameters[param.name] ?? param.default) as string | number | readonly string[] | undefined}
                  onChange={(e) => onChange({
                    ...config,
                    hyperparameters: { ...config.hyperparameters, [param.name]: e.target.value }
                  })}
                  className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                >
                  {param.options?.map((opt: { label: string; value: unknown }) => (
                    <option key={String(opt.value)} value={String(opt.value)}>{opt.label}</option>
                  ))}
                </select>
              ) : (
                <HyperparameterInput
                  type={param.type}
                  value={config.hyperparameters[param.name] ?? param.default}
                  onChange={(val) => onChange({
                    ...config,
                    hyperparameters: { ...config.hyperparameters, [param.name]: val }
                  })}
                  step={param.step}
                  min={param.min}
                  max={param.max}
                />
              )}
            </div>
          ))
        )}
        {hyperparameters.length === 0 && !isLoadingDefs && (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic text-center py-4">
            No parameters available.
          </p>
        )}
      </div>
    </div>
  );

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded text-xs text-blue-700 dark:text-blue-300 flex justify-between items-start gap-2">
          <span>Group rows into clusters by similarity — no target column needed.</span>
          <button
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

      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          onClick={() => { void handleTrain(); }}
          disabled={!datasetId}
          title={!datasetId ? 'Connect a dataset node upstream to enable training' : undefined}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-lg disabled:hover:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          <Play className="w-4 h-4 fill-current" />
          <span className="text-sm font-semibold">Start Segmentation</span>
        </button>
      </div>
    </div>
  );
};
