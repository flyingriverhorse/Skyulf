import { useEffect, useId, useState, useRef } from 'react';
import { useValidationReveal } from '../../../../components/shared/ValidationField';
import { jobsApi } from '../../../../core/api/jobs';
import { RegistryItem, registryApi } from '../../../../core/api/registry';
import { useIsWideContainer } from '../../../../core/hooks/useIsWideContainer';
import { useTrainingNodeContext } from '../../../../core/hooks/useTrainingNodeContext';
import type { HyperparameterDef } from '../components/types';
import type { TrainingConfig, TrainingTask } from '../TrainingSettings';
import { getTaskForModelType } from '../../../../components/pages/ExperimentsPage/utils/jobMeta';
import { TASK_TAG, isClassificationModel, isTrainingModel, isGridStrategy, hasLoadedSearchSpace } from './modelOptions';

/** Keep inspector state and asynchronous definitions alive when visible tabs change. */
export function useTrainingSettings(
  config: TrainingConfig,
  onChange: (c: TrainingConfig) => void,
  nodeId: string | undefined,
  task: TrainingTask | undefined,
) {
  const isAdvanced = config.run_mode === 'advanced';
  const taskTag = task ? TASK_TAG[task] : undefined;

  // Basic-mode hyperparameter defs/editor state.
  const [hyperparameters, setHyperparameters] = useState<HyperparameterDef[]>([]);
  const [isLoadingHyperparamDefs, setIsLoadingHyperparamDefs] = useState(false);

  // Advanced-mode search-space defs/editor state.
  const [searchSpaceDefs, setSearchSpaceDefs] = useState<HyperparameterDef[]>([]);
  const [isLoadingSearchSpaceDefs, setIsLoadingSearchSpaceDefs] = useState(false);
  const fieldId = useId();
  const [showStrategyModal, setShowStrategyModal] = useState(false);

  const [showParamsModal, setShowParamsModal] = useState(false);
  const [showInfo, setShowInfo] = useState(() => !sessionStorage.getItem('hide_info_training_node'));

  const { availableColumns, upstreamTarget, datasetId, runJob, isSubmitting, submissionMessage, runFeedback } = useTrainingNodeContext(nodeId);

  // Responsive layout: switch to a 2-column layout once the panel is wider than 450px.
  const [containerRef, isWide] = useIsWideContainer();
  const [activeTab, setActiveTab] = useState<'model' | 'params'>('model');
  useValidationReveal((field) => {
    if (field === 'model_type' || field === 'target_column') setActiveTab('model');
  });
  const [showCV, setShowCV] = useState(false);
  const [availableModels, setAvailableModels] = useState<RegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [showScalingAlert, setShowScalingAlert] = useState(true);

  // Find currently selected model item to check tags
  const selectedModelItem = availableModels.find(m => m.id === config.model_type);
  const requiresScaling = selectedModelItem?.tags?.includes('requires_scaling');

  // Classification-only options (e.g. decision-threshold tuning): task-scoped
  // nodes know their task; the generic node falls back to the selected
  // model's registry tags.
  const isClassification = isClassificationModel(task, selectedModelItem);

  // Fetch available models from registry. Shared between both modes — the
  // dedicated Segmentation node has its own model list (see
  // `SegmentationSettings`); this node always excludes clustering algorithms,
  // since its target-column/CV-driven flow doesn't apply to them.
  useEffect(() => {
      const fetchModels = async () => {
          setIsLoadingModels(true);
          try {
              const nodes = await registryApi.getAllNodes();
              // Accept both "Model" (old) and "Modeling" (new skyulf-core).
              const models = nodes.filter(n => isTrainingModel(n, taskTag));
              setAvailableModels(models);
          } catch (error) {
              console.error("Failed to fetch models:", error);
              // Fallback to static list if API fails
              setAvailableModels([
                  { id: 'random_forest_classifier', name: 'Random Forest Classifier', category: 'Modeling', description: '', params: {} },
                  { id: 'logistic_regression', name: 'Logistic Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'sgd_classifier', name: 'SGD Classifier', category: 'Modeling', description: '', params: {} },
                  { id: 'ridge_regression', name: 'Ridge Regression', category: 'Modeling', description: '', params: {} },
                  { id: 'random_forest_regressor', name: 'Random Forest Regressor', category: 'Modeling', description: '', params: {} },
              ]);
          } finally {
              setIsLoadingModels(false);
          }
      };
      fetchModels();
  }, [taskTag]);

  // We use a ref to track if customization was active before model switch (basic mode).
  const keepCustomizationOpen = useRef(false);

  // Basic mode: fetch hyperparameter definitions when model type changes.
  useEffect(() => {
    if (isAdvanced) return;
    if (config.model_type) {
      setIsLoadingHyperparamDefs(true);
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
        .finally(() => { setIsLoadingHyperparamDefs(false); });
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type, isAdvanced]);

  // Advanced mode: fetch search-space definitions/defaults when model or
  // strategy changes. Mirrors the former AdvancedTuningSettings effect.
  const loadedModelTypeRef = useRef<string | null>(null);
  const loadedStrategyRef = useRef<string | null>(null);

  useEffect(() => {
      if (!isAdvanced) return;
      const applySearchDefaults = (defaults: Record<string, unknown>, modelType: string, strategy: string) => {
          onChange({
              ...config,
              search_space: defaults || {}
          });
          loadedModelTypeRef.current = modelType;
          loadedStrategyRef.current = strategy;
      };
      const loadModelData = async () => {
          const modelType = config.model_type;
          const strategy = config.search_strategy ?? 'random';
          if (!modelType) return;

          const isNewModel = modelType !== loadedModelTypeRef.current;
          // Also reload when switching between grid and non-grid strategies
          // so the search space is appropriate for the selected method.
          const wasGrid = isGridStrategy(loadedStrategyRef.current);
          const isGrid = isGridStrategy(strategy);
          const isStrategyClassChange = wasGrid !== isGrid;

          if (hasLoadedSearchSpace(isNewModel, isStrategyClassChange, searchSpaceDefs)) return;

          setIsLoadingSearchSpaceDefs(true);
          try {
              // 1. Fetch Definitions (only needed when model changes)
              if (isNewModel) {
                  const defs = await jobsApi.getHyperparameters(modelType);
                  setSearchSpaceDefs(defs as HyperparameterDef[]);
              }

              // 2. Fetch Defaults when model or strategy class changes
              if (isNewModel || isStrategyClassChange) {
                  const defaults = await jobsApi.getDefaultSearchSpace(modelType, strategy);
                  applySearchDefaults(defaults, modelType, strategy);
              }
          } catch (error) {
              console.error("Failed to load model hyperparameters:", error);
          } finally {
              setIsLoadingSearchSpaceDefs(false);
          }
      };

      void loadModelData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.model_type, config.search_strategy, isAdvanced]);

  // Auto-select target column from upstream
  useEffect(() => {
    if (upstreamTarget && config.target_column !== upstreamTarget) {
        if (isAdvanced) {
            onChange({ ...config, target_column: upstreamTarget });
        } else if (!config.target_column) {
            // Basic mode only auto-selects when nothing is set yet — matches
            // the former BasicTrainingSettings behavior (advanced mode always
            // synced, basic mode preserved a manual choice).
            onChange({ ...config, target_column: upstreamTarget });
        }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [upstreamTarget, config.target_column, isAdvanced]);

  const resolvedTask = task ?? getTaskForModelType(config.model_type, availableModels);
  const historyTask = resolvedTask === 'other' ? 'classification' : resolvedTask;
  const handleSubmit = async () => {
    // Task-scoped nodes already know their task; the generic (hidden)
    // TrainingNode resolves it from the selected model's registry tags —
    // reuses the same registry fetch as the model dropdown, no extra fetch.
    // Falls back to 'classification' (like jobMeta.ts's own default) if
    // unresolvable — this only affects which drawer tab opens, never
    // whether the job itself is submitted.
    await runJob(isAdvanced ? 'tuning' : 'training', historyTask);
  };

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
  return {
    config,
    onChange,
    isAdvanced,
    hyperparameters,
    isLoadingHyperparamDefs,
    searchSpaceDefs,
    isLoadingSearchSpaceDefs,
    fieldId,
    showStrategyModal,
    setShowStrategyModal,
    showParamsModal,
    setShowParamsModal,
    showInfo,
    setShowInfo,
    availableColumns,
    datasetId,
    isSubmitting,
    submissionMessage,
    runFeedback,
    containerRef,
    isWide,
    activeTab,
    setActiveTab,
    showCV,
    setShowCV,
    availableModels,
    isLoadingModels,
    showScalingAlert,
    setShowScalingAlert,
    selectedModelItem,
    requiresScaling,
    isClassification,
    keepCustomizationOpen,
    historyTask,
    handleSubmit,
    useCustomParams,
    toggleCustomParams,
  };
}

export type TrainingSettingsState = ReturnType<typeof useTrainingSettings>;
