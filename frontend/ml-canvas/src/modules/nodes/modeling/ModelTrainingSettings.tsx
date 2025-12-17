import React, { useEffect, useState, useCallback } from 'react';
import { 
    Play, Download, Loader2, Settings2, Database, Activity, 
    BarChart3, X, Check, ChevronRight, ChevronDown, HelpCircle, AlertCircle, RefreshCw, AlertTriangle
} from 'lucide-react';
import { jobsApi, JobInfo } from '../../../core/api/jobs';
import { useUpstreamData } from '../../../core/hooks/useUpstreamData';
import { useDatasetSchema } from '../../../core/hooks/useDatasetSchema';
import { useElementSize } from '../../../core/hooks/useElementSize';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { convertGraphToPipelineConfig } from '../../../core/utils/pipelineConverter';
import { getIncomers } from '@xyflow/react';

export interface ModelTrainingConfig {
  target_column: string;
  model_type: string;
  hyperparameters: Record<string, unknown>;
  cv_enabled: boolean;
  cv_folds: number;
  cv_type: string;
  cv_shuffle: boolean;
  cv_random_state: number;
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
    step?: number;
}

const Tooltip: React.FC<{ text: string }> = ({ text }) => (
    <div className="group relative flex items-center">
        <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block w-48 p-2 bg-gray-900 text-white text-xs rounded shadow-lg z-50">
            {text}
            <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
        </div>
    </div>
);

const BestParamsModal: React.FC<{ 
    isOpen: boolean; 
    onClose: () => void; 
    modelType: string; 
    onSelect: (params: unknown) => void;
}> = ({ isOpen, onClose, modelType: initialModelType, onSelect }) => {
    const [currentModelType, setCurrentModelType] = useState(initialModelType);
    const [jobs, setJobs] = useState<JobInfo[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Update currentModelType when initialModelType changes
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
                        <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                            <Activity className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                        </div>
                        <div>
                            <h3 className="font-semibold text-gray-900 dark:text-gray-100">Best Parameters History</h3>
                            <div className="flex items-center gap-2 mt-1">
                                <span className="text-xs text-gray-500 dark:text-gray-400">Select parameters for:</span>
                                <select 
                                    value={currentModelType}
                                    onChange={(e) => { setCurrentModelType(e.target.value); }}
                                    className="text-xs border border-gray-200 dark:border-gray-700 rounded px-2 py-0.5 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-1 focus:ring-blue-500 outline-none"
                                >
                                    <option value="random_forest_classifier">Random Forest Classifier</option>
                                    <option value="logistic_regression">Logistic Regression</option>
                                    <option value="ridge_regression">Ridge Regression</option>
                                    <option value="random_forest_regressor">Random Forest Regressor</option>
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
                            <Loader2 className="w-8 h-8 animate-spin mb-2 text-blue-500" />
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
                            <div key={job.job_id} className="group bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-4 hover:shadow-md hover:border-blue-200 dark:hover:border-blue-800 transition-all">
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
                                                    <span className="text-xs font-medium text-gray-500">Score:</span>
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
                                    <button 
                                        onClick={() => {
                                            if (job.result?.best_params) {
                                                onSelect({
                                                    params: job.result.best_params,
                                                    modelType: currentModelType
                                                });
                                                onClose();
                                            }
                                        }}
                                        className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md transition-colors flex items-center gap-1 shadow-sm shadow-blue-500/20"
                                    >
                                        <Check className="w-3 h-3" />
                                        Apply
                                    </button>
                                </div>
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

const HyperparameterInput: React.FC<{
    value: unknown;
    type: string;
    onChange: (value: unknown) => void;
    step?: number;
    min?: number;
    max?: number;
}> = ({ value, type, onChange, step, min, max }) => {
    const [localValue, setLocalValue] = useState<string>('');

    useEffect(() => {
        setLocalValue(value === null ? 'None' : value?.toString() ?? '');
    }, [value]);

    const handleBlur = () => {
        const trimmed = localValue.trim();
        
        if (trimmed.toLowerCase() === 'none') {
            onChange(null);
            return;
        }

        if (type === 'number') {
            if (trimmed === '') return;
            const num = Number(trimmed);
            if (!isNaN(num)) {
                onChange(num);
            } else {
                // Revert if invalid
                setLocalValue(value === null ? 'None' : value?.toString() ?? '');
            }
        } else if (type === 'boolean') {
            if (trimmed.toLowerCase() === 'true') onChange(true);
            else if (trimmed.toLowerCase() === 'false') onChange(false);
            else setLocalValue(value?.toString() ?? '');
        } else {
            onChange(trimmed);
        }
    };

    return (
        <input
            type="text"
            value={localValue}
            onChange={(e) => { setLocalValue(e.target.value); }}
            onBlur={handleBlur}
            placeholder={value === null ? 'None' : ''}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
            {...(type === 'number' ? { step, min, max } : {})}
        />
    );
};

export const ModelTrainingSettings: React.FC<{ config: ModelTrainingConfig; onChange: (c: ModelTrainingConfig) => void; nodeId?: string }> = ({
  config,
  onChange,
  nodeId,
}) => {
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [showParamsModal, setShowParamsModal] = useState(false);
  const [isLoadingDefs, setIsLoadingDefs] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_model_training'));
  
  const { toggleDrawer, setTab } = useJobStore();
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

  // Fetch hyperparameter definitions when model type changes
  useEffect(() => {
    if (config.model_type) {
      setIsLoadingDefs(true);
      jobsApi.getHyperparameters(config.model_type)
        .then((defs) => setHyperparameters(defs as HyperparameterDef[]))
        .catch(console.error)
        .finally(() => { setIsLoadingDefs(false); });
    }
  }, [config.model_type]);

  // Auto-select target column from upstream
  useEffect(() => {
    const upstreamTarget = upstreamData.find(d => d.target_column)?.target_column as string | undefined;
    if (upstreamTarget && config.target_column !== upstreamTarget) {
        onChange({ ...config, target_column: upstreamTarget });
    }
  }, [upstreamData, config.target_column]);

  const handleTrain = async () => {
    if (!nodeId) return;
    try {
        const pipelineConfig = convertGraphToPipelineConfig(nodes, edges);
        await jobsApi.runPipeline({
            ...pipelineConfig,
            target_node_id: nodeId,
            job_type: 'training'
        });
        alert("Training job submitted successfully!");
        setTab('training');
        toggleDrawer(true);
    } catch (error) {
        console.error("Failed to submit training job:", error);
        alert("Failed to submit training job. Check console for details.");
    }
  };

  const ModelConfigSection = (
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
                                onChange={(e) => onChange({ ...config, model_type: e.target.value, hyperparameters: {} })}
                                className="w-full appearance-none border border-gray-300 dark:border-gray-600 rounded-lg p-2.5 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                            >
                                <option value="random_forest_classifier">Random Forest Classifier</option>
                                <option value="logistic_regression">Logistic Regression</option>
                                <option value="ridge_regression">Ridge Regression</option>
                                <option value="random_forest_regressor">Random Forest Regressor</option>
                            </select>
                            <ChevronDown className="absolute right-3 top-3 w-4 h-4 text-gray-400 pointer-events-none" />
                        </div>
                    </div>

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
                                        </select>
                                    </div>
                                </div>
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

  const HyperparametersSection = () => {
    const useCustomParams = Object.keys(config.hyperparameters).length > 0;

    const toggleCustomParams = (enabled: boolean) => {
        if (enabled) {
            // Initialize with defaults if empty
            const defaults: Record<string, any> = {};
            hyperparameters.forEach(p => {
                defaults[p.name] = p.default;
            });
            onChange({ ...config, hyperparameters: defaults });
        } else {
            onChange({ ...config, hyperparameters: {} });
        }
    };

    return (
    <div className="space-y-4 animate-in fade-in duration-300">
        <div className="flex items-center justify-between">
           <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
             <Settings2 className="w-4 h-4 text-blue-500" />
             Hyperparameters
           </h4>
           <div className="flex items-center gap-2">
               <label className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2 cursor-pointer select-none">
                   <input 
                        type="checkbox" 
                        checked={useCustomParams}
                        onChange={(e) => { toggleCustomParams(e.target.checked); }}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
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
                      onClick={() => setShowParamsModal(true)}
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
                          {param.description && <Tooltip text={param.description} />}
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
  };

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
                <div className="overflow-y-auto pl-2 border-l border-gray-100 dark:border-gray-800"><HyperparametersSection /></div>
            </div>
        ) : (
            <>
                {activeTab === 'model' && ModelConfigSection}
                {activeTab === 'params' && <HyperparametersSection />}
            </>
        )}
      </div>

      {/* Footer */}
      <div className="pt-4 mt-auto border-t border-gray-100 dark:border-gray-700 flex flex-col gap-3 items-center">
        <button
          onClick={handleTrain}
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
