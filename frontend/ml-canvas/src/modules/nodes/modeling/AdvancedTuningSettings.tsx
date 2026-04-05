import React, { useEffect, useState, useRef, useCallback } from 'react';
import { 
    Play, Loader2, Database, Activity, 
    Settings2, BarChart3, X, RefreshCw, ChevronRight, ChevronDown,
    HelpCircle, AlertCircle, AlertTriangle, Check
} from 'lucide-react';
import { jobsApi, JobInfo } from '../../../core/api/jobs';
import { RegistryItem, registryApi } from '../../../core/api/registry';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { formatMetricName } from '../../../core/utils/format';
import { useElementSize } from '../../../core/hooks/useElementSize';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';
import { getIncomers } from '@xyflow/react';
import { StepType } from '../../../core/constants/stepTypes';
import { StrategySettingsModal, StrategyConfig } from './components/StrategySettingsModal';

export interface TuningConfig {
  target_column: string;
  model_type: string;
  search_space: Record<string, unknown>;
  n_trials: number;
  metric: string;
  search_strategy: string;
  strategy_params?: Record<string, unknown>;
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
  random_state: number;
  cv_time_column?: string;
  execution_mode?: 'merge' | 'parallel';
}

interface HyperparameterDef {
    name: string;
    label: string;
    type: 'number' | 'select' | 'boolean';
    default: unknown;
    description?: string;
    options?: { label: string; value: unknown }[];
    min?: number;
    max?: number;
}

const Tooltip: React.FC<{ text: string }> = ({ text }) => (
    <div className="group relative flex items-center">
        <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
        <div className="absolute top-full mt-2 -left-20 hidden group-hover:block w-56 p-2.5 bg-gray-900 text-white text-xs rounded-md shadow-xl z-50">
            {text}
            <div className="absolute bottom-full left-20 ml-1.5 border-4 border-transparent border-b-gray-900" />
        </div>
    </div>
);

const SearchSpaceInput: React.FC<{
    def: HyperparameterDef;
    value: unknown[];
    onChange: (values: unknown[]) => void;
}> = ({ def, value, onChange }) => {
    const [localValue, setLocalValue] = useState('');
    const [error, setError] = useState<string | null>(null);

    // Sync local state with props when props change
    useEffect(() => {
        setLocalValue(Array.isArray(value) ? value.map(v => v === null ? 'None' : v).join(', ') : '');
    }, [value]);

    const validateAndParse = (input: string) => {
        if (!input.trim()) return [];
        
        const parts = input.split(',').map(s => s.trim()).filter(s => s !== '');
        const parsed: unknown[] = [];
        
        for (const part of parts) {
            if (part.toLowerCase() === 'none') {
                parsed.push(null);
                continue;
            }

            if (def.type === 'number') {
                const num = Number(part);
                if (isNaN(num)) {
                    throw new Error(`"${part}" is not a valid number`);
                }
                parsed.push(num);
            } else if (def.type === 'boolean') {
                const lower = part.toLowerCase();
                if (lower === 'true') parsed.push(true);
                else if (lower === 'false') parsed.push(false);
                else throw new Error(`"${part}" must be true or false`);
            } else {
                parsed.push(part);
            }
        }
        return parsed;
    };

    const handleBlur = () => {
        try {
            const parsed = validateAndParse(localValue);
            setError(null);
            // Only trigger change if values actually changed to avoid loops
            if (JSON.stringify(parsed) !== JSON.stringify(value)) {
                onChange(parsed);
            }
        } catch (err: unknown) {
            setError((err as Error).message);
        }
    };

    return (
        <div className="space-y-1">
            <div className="flex justify-between items-center">
                <label className="text-xs font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1">
                    {def.label}
                    {def.description && <Tooltip text={def.description} />}
                </label>
                <span className="text-[10px] text-gray-400 uppercase">{def.type}</span>
            </div>
            
            <div className="relative">
                <input 
                    type="text" 
                    className={`w-full border rounded px-2 py-1.5 text-sm font-mono bg-white dark:bg-gray-900 dark:text-gray-100 outline-none transition-all ${
                        error 
                            ? 'border-red-500 focus:ring-2 focus:ring-red-500/20' 
                            : 'border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500'
                    }`}
                    placeholder={def.type === 'select' ? "e.g. lbfgs, liblinear" : "e.g. 10, 50, 100"}
                    value={localValue}
                    onChange={(e) => {
                        setLocalValue(e.target.value);
                        setError(null); // Clear error while typing
                    }}
                    onBlur={handleBlur}
                />
                {def.type === 'select' && def.options && (
                    <div className="mt-1 flex flex-wrap gap-1">
                        {def.options.map(opt => (
                            <button
                                key={String(opt.value)}
                                onClick={() => {
                                    // Toggle option
                                    const current = validateAndParse(localValue);
                                    const exists = current.includes(opt.value);
                                    const newValue = exists 
                                        ? current.filter(v => v !== opt.value)
                                        : [...current, opt.value];
                                    onChange(newValue);
                                }}
                                className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                                    value.includes(opt.value)
                                        ? 'bg-purple-100 border-purple-200 text-purple-700 dark:bg-purple-900/30 dark:border-purple-800 dark:text-purple-300'
                                        : 'bg-gray-50 border-gray-200 text-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                }`}
                            >
                                {opt.label}
                            </button>
                        ))}
                    </div>
                )}
            </div>
            {error && (
                <p className="text-[10px] text-red-500 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    {error}
                </p>
            )}
        </div>
    );
};

const BestParamsModal: React.FC<{ 
    isOpen: boolean; 
    onClose: () => void; 
    modelType: string;
    availableModels?: RegistryItem[];
}> = ({ isOpen, onClose, modelType: initialModelType, availableModels = [] }) => {
    const [currentModelType, setCurrentModelType] = useState(initialModelType);
    const [jobs, setJobs] = useState<JobInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Update currentModelType when initialModelType changes (e.g. when opening modal)
    useEffect(() => {
        if (isOpen) {
            setCurrentModelType(initialModelType);
        }
    }, [isOpen, initialModelType]);

    const fetchJobs = useCallback(() => {
        if (!currentModelType) return;
        setIsLoading(true);
        setError(null);
        // Use getTuningHistory instead of getTuningJobsForModel
        jobsApi.getTuningHistory(currentModelType)
            .then(data => {
                setJobs(data);
            })
            .catch(err => {
                console.error("Failed to fetch jobs", err);
                setError("Failed to load history.");
            })
            .finally(() => { setIsLoading(false); });
    }, [currentModelType]);

    useEffect(() => {
        if (isOpen) {
            fetchJobs();
        }
    }, [isOpen, fetchJobs]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex justify-center items-center p-4">
            <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
            <div className="relative w-full max-w-2xl max-h-[85vh] bg-white dark:bg-gray-800 shadow-2xl rounded-xl flex flex-col border border-gray-200 dark:border-gray-700 overflow-hidden animate-in fade-in zoom-in duration-200">
                {/* Header */}
                <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50/50 dark:bg-gray-800/50">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
                            <Activity className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Best Parameters History</h3>
                            <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs text-gray-500 dark:text-gray-400">View parameters for:</span>
                                <select 
                                    value={currentModelType}
                                    onChange={(e) => { setCurrentModelType(e.target.value); }}
                                    className="text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-0.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-1 focus:ring-purple-500 outline-none"
                                >
                                    {availableModels.length > 0 ? (
                                        availableModels.map(model => (
                                            <option key={model.id} value={model.id}>{model.name}</option>
                                        ))
                                    ) : (
                                        <>
                                            <option value="random_forest_classifier">Random Forest Classifier</option>
                                            <option value="logistic_regression">Logistic Regression</option>
                                            <option value="ridge_regression">Ridge Regression</option>
                                            <option value="random_forest_regressor">Random Forest Regressor</option>
                                        </>
                                    )}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <button 
                            onClick={fetchJobs}
                            className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 transition-colors"
                            title="Refresh"
                        >
                            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                        </button>
                        <button onClick={onClose} className="p-2 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 transition-colors">
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                
                {/* Content */}
                <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50/30 dark:bg-gray-900/30">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                            <Loader2 className="w-8 h-8 animate-spin mb-2 text-purple-500" />
                            <p className="text-sm">Loading history...</p>
                        </div>
                    ) : error ? (
                        <div className="text-center py-12 text-red-500 bg-red-50 dark:bg-red-900/10 rounded-lg border border-red-100 dark:border-red-900/20">
                            <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                            <p className="text-sm">{error}</p>
                            <button onClick={fetchJobs} className="mt-2 text-xs underline hover:text-red-600">Try Again</button>
                        </div>
                    ) : jobs.length === 0 ? (
                        <div className="text-center py-12 text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-700">
                            <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                            <p className="text-sm">No completed optimization jobs found.</p>
                            <p className="text-xs opacity-70 mt-1">Run an optimization job to see results here.</p>
                        </div>
                    ) : (
                        jobs.map(job => (
                            <div key={job.job_id} className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 hover:shadow-md hover:border-purple-200 dark:hover:border-purple-800 transition-all">
                                <div className="flex justify-between items-start mb-3">
                                    <div className="space-y-1">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs font-mono bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded text-gray-600 dark:text-gray-300">
                                                #{job.job_id.slice(0, 8)}
                                            </span>
                                            <span className="text-xs text-gray-400">
                                                {job.end_time ? new Date(job.end_time).toLocaleString() : 'Unknown Date'}
                                            </span>
                                        </div>
                                        <div className="flex items-center gap-3 mt-1">
                                            {typeof job.result?.best_score === 'number' && (
                                                <div className="flex items-center gap-1">
                                                    <span className="text-xs font-medium text-gray-500">
                                                        {formatMetricName((job.result as Record<string, unknown>).scoring_metric as string) || 'Score'}:
                                                    </span>
                                                    <span className="text-sm font-bold text-green-600 dark:text-green-400">
                                                        {job.result.best_score.toFixed(4)}
                                                    </span>
                                                </div>
                                            )}
                                            <span className="text-[10px] px-1.5 py-0.5 bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full flex items-center gap-1">
                                                <Check className="w-3 h-3" /> Model Ready
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export const AdvancedTuningSettings: React.FC<{ config: TuningConfig; onChange: (c: TuningConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const [searchSpaceDefs, setSearchSpaceDefs] = useState<HyperparameterDef[]>([]);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [showParamsModal, setShowParamsModal] = useState(false);
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'config' | 'search_space'>('config');
  const [showCV, setShowCV] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_model_optimizer'));
  
  const [containerRef, { width }] = useElementSize();
  const isWide = width > 450;

  // --- Upstream Data Logic ---
  const { toggleDrawer, setTab, setActiveParallelRun, startPolling } = useJobStore();
  const upstreamData = useUpstreamData(nodeId || '');
  const nodes = useGraphStore((state) => state.nodes);
  const edges = useGraphStore((state) => state.edges);

  const findUpstreamDatasetId = (currentNodeId: string): string | undefined => {
    const visited = new Set<string>();
    const queue = [currentNodeId];
    while (queue.length > 0) {
      const id = queue.shift()!;
      if (visited.has(id)) continue;
      visited.add(id);
      const node = nodes.find(n => n.id === id);
      if (!node) continue;
      if (id !== currentNodeId) {
        if (node.data?.datasetId) return node.data.datasetId as string;
        if (node.data?.dataset_id) return node.data.dataset_id as string;
        // Check config/params
        const anyData = node.data as unknown;
        if (((anyData as Record<string, unknown>).config as Record<string, unknown>)?.datasetId) return ((anyData as Record<string, unknown>).config as Record<string, unknown>).datasetId as string;
        if (((anyData as Record<string, unknown>).config as Record<string, unknown>)?.dataset_id) return ((anyData as Record<string, unknown>).config as Record<string, unknown>).dataset_id as string;
      }
      const incomers = getIncomers(node, nodes, edges);
      for (const incomer of incomers) queue.push(incomer.id);
    }
    return undefined;
  };

  const datasetId = findUpstreamDatasetId(nodeId || '');
  const { data: schema } = useDatasetSchema(datasetId);
  const availableColumns = schema ? Object.values(schema.columns) : [];
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showScalingAlert, setShowScalingAlert] = useState(true);

  const selectedModelItem = availableModels.find(m => m.id === config.model_type);

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
              // Fallback
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

  // Auto-select target
  useEffect(() => {
    const upstreamTarget = upstreamData.find(d => d.target_column)?.target_column as string | undefined;
    if (upstreamTarget && config.target_column !== upstreamTarget) {
        onChange({ ...config, target_column: upstreamTarget });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamData, config.target_column]);

  // --- Model & Search Space Logic ---

  // We use a ref to track the last model type we successfully loaded defaults for
  // This prevents infinite loops and ensures we only load defaults when the user actually changes the model
  const loadedModelTypeRef = useRef<string | null>(null);

  useEffect(() => {
      const loadModelData = async () => {
          const modelType = config.model_type;
          if (!modelType) return;

          // If we already loaded data for this model type, just ensure defs are present
          // But if the user switched models (config.model_type !== loadedModelTypeRef.current), we MUST reload defaults
          const isNewModel = modelType !== loadedModelTypeRef.current;

          if (!isNewModel && searchSpaceDefs.length > 0) return;

          setIsLoadingDefs(true);
          try {
              // 1. Fetch Definitions
              const defs = await jobsApi.getHyperparameters(modelType);
              setSearchSpaceDefs(defs as HyperparameterDef[]);

              // 2. Fetch Defaults ONLY if it's a new model selection
              if (isNewModel) {
                  const defaults = await jobsApi.getDefaultSearchSpace(modelType);
                  
                  // Update config with new defaults
                  // We use a functional update to ensure we don't lose other concurrent changes
                  // BUT since we are calling onChange from parent prop, we just pass the new object
                  onChange({
                      ...config,
                      search_space: defaults || {}
                  });
                  
                  loadedModelTypeRef.current = modelType;
              }
          } catch (error) {
              console.error("Failed to load model hyperparameters:", error);
          } finally {
              setIsLoadingDefs(false);
          }
      };

      void loadModelData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type]);


  const handleTune = async () => {
    if (!nodeId) return;
    try {
        const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
        const response = await jobsApi.runPipeline({
            ...pipelineConfig,
            target_node_id: nodeId,
            job_type: StepType.ADVANCED_TUNING
        });
        const jobCount = response.job_ids?.length ?? 1;
        if (jobCount > 1) {
            setActiveParallelRun({ jobIds: response.job_ids, startedAt: new Date().toISOString() });
            startPolling();
            alert(`Parallel execution started! ${jobCount} branches submitted.`);
        } else {
            alert("Training job submitted successfully!");
        }
        setTab('advanced_tuning');
        toggleDrawer(true);
    } catch (error) {
        console.error("Failed to submit tuning job:", error);
        alert("Failed to submit tuning job.");
    }
  };

  // --- Render Sections ---

  const configSectionContent = (
    <div className="space-y-5 animate-in fade-in duration-300">
        {/* Model & Target */}
        <div className="space-y-4">
            <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Model Configuration</label>
                <div className="grid gap-3">
                    <div>
                        <label className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Model Type</label>
                        <div className="relative">
                            <select
                                value={config.model_type}
                                onChange={(e) => {
                                    // We don't reset loadedModelTypeRef here anymore, 
                                    // the useEffect will handle the change detection.
                                    onChange({ ...config, model_type: e.target.value, search_space: {} });
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
                    </div>

                   {selectedModelItem?.tags?.includes('requires_scaling') && (
                        <div className="mt-2 text-xs border border-blue-200 dark:border-blue-800 rounded-md bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 overflow-hidden transition-all">
                            <button 
                                onClick={() => setShowScalingAlert(!showScalingAlert)}
                                className="w-full flex items-center justify-between p-2 hover:bg-blue-100 dark:hover:bg-blue-900/40 transition-colors"
                            >
                                <div className="flex items-center gap-2 font-semibold">
                                    <Settings2 className="w-3 h-3" />
                                    <span>Scale Your Data</span>
                                </div>
                                <ChevronDown className={`w-3 h-3 transition-transform ${showScalingAlert ? 'rotate-180' : ''}`} />
                            </button>
                            {showScalingAlert && (
                                <div className="p-2 pt-0 opacity-90 animate-in slide-in-from-top-1 pl-7">
                                    <p className="opacity-90">
                                        This model performs best with scaled features. Consider adding a 
                                        <span className="font-mono mx-1 bg-white dark:bg-black px-1 rounded border border-blue-200 dark:border-blue-800">Scaler</span> 
                                        node before this step.
                                    </p>
                                </div>
                            )}
                        </div>
                   )}

                    <div>
                        <label className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Target Column</label>
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
                                    className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition-all"
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

            {/* Strategy & Metrics */}
            <div className="space-y-1.5">
                <label className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">Tuning Strategy</label>
                <div className="grid grid-cols-2 gap-3">
                    <div className="col-span-2">
                      <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-1.5">
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300">
                                    Search Method
                                </label>
                                <Tooltip text={
                                    config.search_strategy === 'optuna' ? 'Optuna uses Bayesian optimization (TPE) to efficiently find optimal hyperparameters with early pruning.' :
                                    config.search_strategy === 'halving_grid' ? 'Successive Halving (Grid) tests all combinations but quickly drops poorly performing candidates to save time.' :
                                    config.search_strategy === 'halving_random' ? 'Successive Halving (Random) tests random combinations but quickly drops poorly performing candidates.' :
                                    config.search_strategy === 'grid' ? 'Grid Search tests every single combination in the search space. Can be very slow and computationally expensive.' :
                                    'Random Search tests a random subset of parameter combinations. Fast and often surprisingly effective compared to Grid Search.'
                                } />
                            </div>
                          {(config.search_strategy === 'halving_grid' || config.search_strategy === 'halving_random' || config.search_strategy === 'optuna') && (
                              <button
                                  type="button"
                                  onClick={() => setShowStrategyModal(true)}
                                  className="text-blue-600 hover:text-blue-700 dark:text-blue-400 p-1 rounded hover:bg-blue-50 dark:hover:bg-blue-900/20 transition group flex items-center justify-center"
                                  title={`${config.search_strategy.replace('_', ' ')} Settings`}
                              >
                                  <Settings2 size={14} className="group-hover:rotate-45 transition-transform duration-300" />
                              </button>
                          )}
                      </div>
                      <select
                          value={config.search_strategy ?? 'random'}
                          onChange={(e) => {
                              const newStrategy = e.target.value;
                              // Clear strategy params when switching to plain grid/random to avoid passing junk
                              const newParams = (newStrategy === 'grid' || newStrategy === 'random') 
                                                ? {} 
                                                : config.strategy_params || {};
                              onChange({ 
                                  ...config, 
                                  search_strategy: newStrategy,
                                  strategy_params: newParams 
                              });
                          }}
                            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                        >
                            <option value="random">Random Search</option>
                            <option value="grid">Grid Search</option>
                            <option value="halving_grid">Successive Halving (Grid)</option>
                            <option value="halving_random">Successive Halving (Randomized)</option>
                            <option value="optuna">Optuna Search</option>
                        </select>
                    </div>
                    
                    <div>
                        <label className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Metric</label>
                        <select
                            value={config.metric}
                            onChange={(e) => onChange({ ...config, metric: e.target.value })}
                            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                        >
                            <option value="accuracy">Accuracy</option>
                            <option value="f1">F1 Score</option>
                            <option value="roc_auc">ROC AUC</option>
                            <option value="mse">MSE</option>
                            <option value="rmse">RMSE</option>
                            <option value="mae">MAE</option>
                            <option value="r2">R2 Score</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-xs font-medium mb-1 text-gray-700 dark:text-gray-300">Trials</label>
                        <input
                            type="number"
                            value={config.n_trials}
                            onChange={(e) => onChange({ ...config, n_trials: Number(e.target.value) })}
                            disabled={['grid', 'halving_grid'].includes(config.search_strategy)}
                            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 disabled:opacity-50"
                            min={1}
                        />
                    </div>
                </div>
            </div>

            {/* Cross Validation Toggle */}
            <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                <button 
                    onClick={() => { setShowCV(!showCV); }}
                    className="w-full flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                    <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-purple-500" />
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
                                className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                            />
                            <label htmlFor="cv_enabled" className="text-sm text-gray-700 dark:text-gray-300">Enable Cross-Validation</label>
                        </div>

                        {config.cv_enabled !== false && (
                            <div className="space-y-3 pl-6 border-l-2 border-gray-100 dark:border-gray-800">
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-xs text-gray-500 mb-1">Folds</label>
                                        <input
                                            type="number"
                                            value={config.cv_folds ?? 5}
                                            onChange={(e) => onChange({ ...config, cv_folds: Number(e.target.value) })}
                                            className="w-full border border-gray-300 dark:border-gray-600 rounded p-1.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100"
                                            min={2}
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs text-gray-500 mb-1">Method</label>
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
                                            <label className="block text-xs text-gray-500 mb-1">Time Column (optional)</label>
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
                                        className="rounded border-gray-300 text-purple-600 focus:ring-purple-500"
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

  const searchSpaceSectionContent = (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <Settings2 className="w-4 h-4 text-purple-500" />
                Hyperparameters
            </h4>
        </div>

        {isLoadingDefs ? (
            <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
            </div>
        ) : (
            <div className="space-y-3">
                {searchSpaceDefs.map(def => (
                    <div key={`${config.model_type}-${def.name}`} className="bg-gray-50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                        <SearchSpaceInput
                            def={def}
                            value={(config.search_space?.[def.name] || []) as unknown[]}
                            onChange={(newValues) => {
                                onChange({
                                    ...config,
                                    search_space: {
                                        ...config.search_space,
                                        [def.name]: newValues
                                    }
                                });
                            }}
                        />
                    </div>
                ))}
                {searchSpaceDefs.length === 0 && (
                    <div className="text-center py-8 text-gray-500 text-sm">
                        No hyperparameters available for this model.
                    </div>
                )}
            </div>
        )}
    </div>
  );

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {showInfo && (
        <div className="mb-4 p-2 bg-purple-50 dark:bg-purple-900/20 border border-purple-100 dark:border-purple-800 rounded text-xs text-purple-700 dark:text-purple-300 flex justify-between items-start gap-2">
          <span>Automatically explore configurations to find the best performing model.</span>
          <button 
            onClick={() => {
                setShowInfo(false);
                sessionStorage.setItem('hide_info_model_optimizer', 'true');
            }} 
            className="text-purple-400 hover:text-purple-600 dark:hover:text-purple-200"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Mobile/Narrow Tabs */}
      {!isWide && (
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-4">
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'config' 
                ? 'border-purple-500 text-purple-600 dark:text-purple-400' 
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            onClick={() => { setActiveTab('config'); }}
          >
            Configuration
          </button>
          <button
            className={`flex-1 py-2.5 text-xs font-medium text-center border-b-2 transition-colors ${
              activeTab === 'search_space' 
                ? 'border-purple-500 text-purple-600 dark:text-purple-400' 
                : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700'
            }`}
            onClick={() => { setActiveTab('search_space'); }}
          >
            Search Space
          </button>
        </div>
      )}

      <div className="flex-1 overflow-y-auto px-1 pb-4 custom-scrollbar">
        {isWide ? (
            <div className="grid grid-cols-2 gap-6 h-full">
                <div className="overflow-y-auto pr-2">{configSectionContent}</div>
                <div className="overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-800">{searchSpaceSectionContent}</div>
            </div>
        ) : (
            <>
                {activeTab === 'config' && configSectionContent}
                {activeTab === 'search_space' && searchSpaceSectionContent}
            </>
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        {/* Parallel Execution Toggle — only visible when multiple inputs feed this node */}
        {nodeId && new Set(edges.filter(e => e.target === nodeId).map(e => e.source)).size > 1 && (
          <div className="w-full max-w-xs flex items-center justify-between px-3 py-2 rounded-lg bg-slate-50 dark:bg-slate-800/50 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Multi-Input Mode</span>
              <Tooltip text="Merge combines all inputs into one dataset. Parallel runs each input as a separate experiment." />
            </div>
            <div className="flex rounded-md overflow-hidden border border-slate-300 dark:border-slate-600 text-[11px] font-medium">
              <button
                onClick={() => onChange({ ...config, execution_mode: 'merge' })}
                className={`px-2.5 py-1 transition-colors ${
                  (config.execution_mode || 'merge') === 'merge'
                    ? 'bg-purple-500 text-white'
                    : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
                }`}
              >
                Merge
              </button>
              <button
                onClick={() => onChange({ ...config, execution_mode: 'parallel' })}
                className={`px-2.5 py-1 transition-colors ${
                  config.execution_mode === 'parallel'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white dark:bg-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-600'
                }`}
              >
                Parallel
              </button>
            </div>
          </div>
        )}
        <button
          onClick={() => { void handleTune(); }}
          className="w-full max-w-xs flex items-center justify-center gap-2 px-6 py-2.5 text-white rounded-lg shadow-lg transition-all hover:shadow-xl hover:-translate-y-0.5 active:translate-y-0"
          style={{ background: 'var(--main-gradient)' }}
        >
          <Play className="w-4 h-4 fill-current" />
          <span className="text-sm font-semibold">Start Advanced Training</span>
        </button>

        <button 
            onClick={() => { setShowParamsModal(true); }}
            className="text-xs text-gray-500 hover:text-purple-600 dark:text-gray-400 dark:hover:text-purple-400 flex items-center gap-1.5 transition-colors px-3 py-1 rounded-md hover:bg-gray-50 dark:hover:bg-gray-800"
        >
            <Activity className="w-3.5 h-3.5" />
            View Best Parameters History
        </button>
      </div>

      <BestParamsModal
        isOpen={showParamsModal}
        onClose={() => { setShowParamsModal(false); }}
        modelType={config.model_type}
        availableModels={availableModels}
      />

      <StrategySettingsModal
        isOpen={showStrategyModal}
        onClose={() => setShowStrategyModal(false)}
        strategy={config.search_strategy || 'random'}
        initialConfig={config.strategy_params as StrategyConfig | undefined}
        onSave={(newStrategyParams) => {
            onChange({ ...config, strategy_params: newStrategyParams as Record<string, unknown> });
        }}
      />
    </div>
  );
};
