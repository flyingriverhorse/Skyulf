import React, { useEffect, useState, useRef } from 'react';
import {
    Play, Download, Loader2, Settings2,
    BarChart3, X, ChevronRight, ChevronDown, AlertCircle, AlertTriangle
} from 'lucide-react';
import { jobsApi } from '../../../core/api/jobs';
import { RegistryItem, registryApi } from '../../../core/api/registry';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useElementSize } from '../../../core/hooks/useElementSize';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';
import { getIncomers } from '@xyflow/react';
import { StepType } from '../../../core/constants/stepTypes';
import { HelpTooltip } from './components/HelpTooltip';
import { BestParamsModal } from './components/BestParamsModal';
import { HyperparameterInput } from './components/HyperparameterInput';
import type { HyperparameterDef } from './components/types';
import type { ExecutionMode } from '../../../core/types/executionMode';
import { toast } from '../../../core/toast';

export interface ModelTrainingConfig {
  target_column: string;
  model_type: string;
  hyperparameters: Record<string, unknown>;
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
  cv_time_column?: string;
  execution_mode?: ExecutionMode;
}

// HyperparameterDef, HelpTooltip, BestParamsModal, and HyperparameterInput
// were extracted into ./components/* to keep this file focused on the panel.

export const BasicTrainingSettings: React.FC<{ config: ModelTrainingConfig; onChange: (c: ModelTrainingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [showParamsModal, setShowParamsModal] = useState(false);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_model_training'));
  
  const { toggleDrawer: toggleJobDrawer, setTab, setActiveParallelRun, startPolling } = useJobStore();
  const upstreamData = useUpstreamData(nodeId || '');
  
  // Recursive search for datasetId
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
      
      // If this is NOT the current node, check if it has datasetId
      if (id !== currentNodeId) {
        if (node.data?.datasetId) return node.data.datasetId as string;
        if (node.data?.dataset_id) return node.data.dataset_id as string;
        
        if (node.data?.config) {
            const config = node.data.config as Record<string, unknown>;
            if (config.datasetId) return config.datasetId as string;
            if (config.dataset_id) return config.dataset_id as string;
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

  const [containerRef, { width }] = useElementSize();
  const isWide = width > 450; 
  const [activeTab, setActiveTab] = useState<'model' | 'params'>('model');
  const [showCV, setShowCV] = useState(false);
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showScalingAlert, setShowScalingAlert] = useState(true);
  
  // Find currently selected model item to check tags
  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');

  // Fetch available models from registry
  useEffect(() => {
      const fetchModels = async () => {
          setIsLoadingModels(true);
          try {
              const nodes = await registryApi.getAllNodes();
              // Accept both "Model" (old) and "Modeling" (new skyulf-core)
              const models = nodes.filter(n => n.category === 'Model' || n.category === 'Modeling');
              setAvailableModels(models);
          } catch (error) {
              console.error("Failed to fetch models:", error);
              // Fallback to static list if API fails
              setAvailableModels([
                  { id: 'random_forest_classifier', name: 'Random Forest Classifier', category: 'Modeling', description: '', params: {} },
                  { id: 'logistic_regression', name: 'Logistic Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'ridge_regression', name: 'Ridge Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'random_forest_regressor', name: 'Random Forest Regressor', category: 'Modeling', description: '', params: {} },
              ]);
          } finally {
              setIsLoadingModels(false);
          }
      };
      fetchModels();
  }, []);

  // We use a ref to track if customization was active before model switch
  const keepCustomizationOpen = useRef(false);

  // Fetch hyperparameter definitions when model type changes
  useEffect(() => {
    if (config.model_type) {
      setIsLoadingDefs(true);
      jobsApi.getHyperparameters(config.model_type)
        .then((defs) => {
            const definitions = defs as HyperparameterDef[];
            setHyperparameters(definitions);

            // If we switched models while customization was active, apply new defaults immediately
            if (keepCustomizationOpen.current) {
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

  // Auto-select target column from upstream
  useEffect(() => {
    const upstreamTarget = upstreamData.find(d => d.target_column)?.target_column as string | undefined;
    // Only update if we have a new upstream target AND it's different from current
    // AND we haven't manually set one (optional heuristic, but for now just check difference)
    if (upstreamTarget && config.target_column !== upstreamTarget) {
        // Check if the current target is valid (in available columns)
        // If current target is empty, auto-select.
        if (!config.target_column) {
             onChange({ ...config, target_column: upstreamTarget });
        }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamData, config.target_column, onChange]);

  const handleTrain = async () => {
    if (!nodeId) return;
    try {
        const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
        const response = await jobsApi.runPipeline({
            ...pipelineConfig,
            target_node_id: nodeId,
            job_type: StepType.BASIC_TRAINING
        });
        const jobCount = response.job_ids?.length || 1;
        if (jobCount > 1) {
            setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
            startPolling();
            toast.success(`Parallel execution started`, `${jobCount} branches submitted.`);
        } else {
            toast.success('Training job submitted');
        }
        setTab('basic_training');
        toggleJobDrawer(true);
    } catch (error) {
        console.error("Failed to submit training job:", error);
        toast.error('Failed to submit training job', 'Check console for details.');
    }
  };

  const ModelConfigSection = (
    <div className="space-y-5 animate-in fade-in duration-300">
        {/* Model & Target */}
        <div className="space-y-4">
            <div className="space-y-1.5">
                <span className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</span>
                <div className="grid gap-3">
                    <div>
                        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Model Type</span>
                        <div className="relative">
                            <select
                                value={config.model_type}
                                onChange={(e) => {
                                    // Check if customization is active before switching
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
                        <span className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</span>
                        {availableColumns.length === 0 && (
                            <div className="mb-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded text-xs text-yellow-700 dark:text-yellow-400 flex items-center gap-2">
                                <AlertTriangle className="w-3 h-3" />
                                <span>Connect a dataset node to see available columns</span>
                            </div>
                        )}
                        <div className="relative">
                            {availableColumns.length > 0 ? (
                                <select
                                    value={config.target_column}
                                    onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                                    className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                                >
                                    <option value="">Select target...</option>
                                    {availableColumns.map((col) => (
                                        <option key={col.name} value={col.name}>{col.name}</option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    type="text"
                                    value={config.target_column}
                                    onChange={(e) => onChange({ ...config, target_column: e.target.value })}
                                    className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                    placeholder="e.g., target"
                                />
                            )}
                            {availableColumns.length > 0 && <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />}
                        </div>
                    </div>
                </div>
            </div>

            <div className="border-t border-gray-100 dark:border-gray-700" />

            {/* CV Settings */}
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <button 
                    onClick={() => { setShowCV(!showCV); }}
                    className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                    <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-blue-500" />
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-200">Cross Validation</span>
                    </div>
                    <ChevronRight className={`w-4 h-4 text-gray-400 transition-transform ${showCV ? 'rotate-90' : ''}`} />
                </button>
                
                {showCV && (
                    <div className="p-3 space-y-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 animate-in slide-in-from-top-2 duration-200">
                        <div className="flex items-center gap-2 mb-2">
                            <input
                                type="checkbox"
                                id="cv_enabled"
                                checked={config.cv_enabled !== false}
                                onChange={(e) => onChange({ ...config, cv_enabled: e.target.checked })}
                                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                            />
                            <label htmlFor="cv_enabled" className="text-sm text-gray-700 dark:text-gray-300">Enable Cross-Validation</label>
                        </div>

                        {config.cv_enabled !== false && (
                            <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <span className="block text-xs text-gray-500 mb-1">Folds</span>
                                        <input
                                            type="number"
                                            value={config.cv_folds ?? 5}
                                            onChange={(e) => onChange({ ...config, cv_folds: Number(e.target.value) })}
                                            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                            min={2}
                                        />
                                    </div>
                                    <div>
                                        <span className="block text-xs text-gray-500 mb-1">Method</span>
                                        <select
                                            value={config.cv_type ?? 'k_fold'}
                                            onChange={(e) => onChange({ ...config, cv_type: e.target.value })}
                                            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                        >
                                            <option value="k_fold">K-Fold</option>
                                            <option value="stratified_k_fold">Stratified</option>
                                            <option value="time_series_split">Time Series</option>
                                            <option value="shuffle_split">Shuffle Split</option>
                                            <option value="nested_cv">Nested CV</option>
                                        </select>
                                    </div>
                                </div>
                                {config.cv_type === 'time_series_split' && (
                                    <div className="space-y-2">
                                        <div className="flex items-start gap-1.5 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs text-amber-700 dark:text-amber-400">
                                            <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                                            <span>Data must be sorted by time. Select a date column below or ensure your data is pre-sorted.</span>
                                        </div>
                                        <div>
                                            <span className="block text-xs text-gray-500 mb-1">Time Column (optional)</span>
                                            <select
                                                value={config.cv_time_column ?? ''}
                                                onChange={(e) => onChange({ ...config, cv_time_column: e.target.value })}
                                                className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                            >
                                                <option value="">Auto-detect</option>
                                                {availableColumns
                                                    .filter((col) => {
                                                        const dt = String(col.dtype).toLowerCase();
                                                        return dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp');
                                                    })
                                                    .map((col) => (
                                                        <option key={col.name} value={col.name}>{col.name}</option>
                                                    ))
                                                }
                                                {availableColumns
                                                    .filter((col) => {
                                                        const dt = String(col.dtype).toLowerCase();
                                                        return !(dt.includes('datetime') || dt.includes('date') || dt.includes('time') || dt.includes('timestamp'));
                                                    })
                                                    .map((col) => (
                                                        <option key={col.name} value={col.name}>{col.name}</option>
                                                    ))
                                                }
                                            </select>
                                        </div>
                                    </div>
                                )}
                                <div className="flex items-center gap-2">
                                    <input
                                        type="checkbox"
                                        id="cv_shuffle"
                                        checked={config.cv_shuffle !== false}
                                        onChange={(e) => onChange({ ...config, cv_shuffle: e.target.checked })}
                                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                                    />
                                    <label htmlFor="cv_shuffle" className="text-xs text-gray-600 dark:text-gray-400">Shuffle Data</label>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    </div>
  );

  // Defining HyperparametersSection as a nested component caused React to
  // remount it on every parent render — losing input focus and re-running its
  // effects. Use a JSX const instead so it shares this component's render tree.
  const useCustomParams = Object.keys(config.hyperparameters).length > 0;

  const toggleCustomParams = (enabled: boolean) => {
      if (enabled) {
          // Initialize with defaults if empty
          const defaults: Record<string, unknown> = {};
          hyperparameters.forEach(p => {
              defaults[p.name] = p.default;
          });
          // Only update if we don't have params already
          if (Object.keys(config.hyperparameters).length === 0) {
              onChange({ ...config, hyperparameters: defaults });
          }
      } else {
          onChange({ ...config, hyperparameters: {} });
      }
  };

  const HyperparametersSection = (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="flex items-center justify-between">
           <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
             <Settings2 className="w-4 h-4 text-blue-500" />
             Hyperparameters
           </h4>
           <div className="flex items-center gap-2">
               <label className={`text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 cursor-pointer select-none ${isLoadingDefs ? 'opacity-50 cursor-not-allowed' : ''}`}>
                   <input 
                        type="checkbox" 
                        checked={useCustomParams}
                        onChange={(e) => { toggleCustomParams(e.target.checked); }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        disabled={isLoadingDefs}
                   />
                   Customize
               </label>
           </div>
        </div>
        
        {!useCustomParams ? (
            <div className="text-center py-8 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-dashed border-gray-200 dark:border-gray-700">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                    Using default hyperparameters.
                </p>
                <button
                    onClick={() => { setShowParamsModal(true); }}
                    className="mt-3 text-xs flex items-center gap-1.5 px-3 py-1.5 mx-auto bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400 rounded-md border border-purple-200 dark:border-purple-800 hover:bg-purple-100 dark:hover:bg-purple-900/40 transition-colors shadow-sm"
                >
                    <Download className="w-3 h-3" />
                    Load Best Params
                </button>
            </div>
        ) : (
            <div className="space-y-3">
              <div className="flex justify-end mb-2">
                   <button
                      onClick={() => { setShowParamsModal(true); }}
                      className="text-xs flex items-center gap-1.5 px-2 py-1 text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 rounded transition-colors"
                    >
                      <Download className="w-3 h-3" />
                      Load Best
                   </button>
              </div>
              
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
        )}
    </div>
  );

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded text-xs text-blue-700 dark:text-blue-300 flex justify-between items-start gap-2">
          <span>Train a model using fixed parameters or defaults.</span>
          <button 
            onClick={() => {
                setShowInfo(false);
                sessionStorage.setItem('hide_info_model_training', 'true');
            }} 
            className="text-blue-400 hover:text-blue-600 dark:hover:text-blue-200"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      <BestParamsModal 
        isOpen={showParamsModal} 
        onClose={() => { setShowParamsModal(false); }}
        modelType={config.model_type}
        availableModels={availableModels}
        onSelect={(result) => {
            // result contains { params, modelType }
            // If model type differs, we update it too
            const res = result as { modelType?: string; params: Record<string, unknown> };
            if (res.modelType && res.modelType !== config.model_type) {
                onChange({ 
                    ...config, 
                    model_type: res.modelType,
                    hyperparameters: res.params 
                });
            } else {
                onChange({ ...config, hyperparameters: res.params });
            }
        }}
      />
      
      {/* Mobile/Narrow Tabs */}
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

      {/* Footer */}
      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          onClick={() => { void handleTrain(); }}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          <Play className="w-4 h-4 fill-current" />
          <span className="text-sm font-semibold">Start Training</span>
        </button>
      </div>
    </div>
  );
};
