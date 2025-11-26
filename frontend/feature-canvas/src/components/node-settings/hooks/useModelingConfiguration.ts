import { useEffect, useMemo } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { ensureArrayOfString } from '../sharedUtils';
import type { FeatureNodeParameterOption } from '../../../api';
import type { FeatureTargetSplitConfig } from '../nodes/modeling/FeatureTargetSplitSection';
import type { TrainTestSplitConfig } from '../nodes/modeling/TrainTestSplitSection';
import type { TrainModelDraftConfig } from '../nodes/modeling/TrainModelDraftSection';
import type { TrainModelRuntimeConfig } from '../nodes/modeling/ModelTrainingSection';
import type { ClassResamplingConfig } from '../nodes/resampling/ClassResamplingSection';
import type { CatalogFlagMap } from './useCatalogFlags';

export type TrainModelCVConfig = {
  enabled: boolean;
  strategy: 'auto' | 'kfold' | 'stratified_kfold';
  folds: number;
  shuffle: boolean;
  randomState: number | null;
  refitStrategy: 'train_only' | 'train_plus_validation';
};

export type UseModelingConfigurationArgs = {
  configState: Record<string, any>;
  setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
  catalogFlags: CatalogFlagMap;
  upstreamTargetColumn: string;
  nodeId: string;
  modelTypeOptions?: FeatureNodeParameterOption[] | null;
};

export type UseModelingConfigurationResult = {
  featureTargetSplitConfig: FeatureTargetSplitConfig | null;
  trainTestSplitConfig: TrainTestSplitConfig | null;
  resamplingConfig: ClassResamplingConfig;
  trainModelDraftConfig: TrainModelDraftConfig | null;
  trainModelRuntimeConfig: TrainModelRuntimeConfig | null;
  trainModelCVConfig: TrainModelCVConfig;
  filteredModelTypeOptions: FeatureNodeParameterOption[];
};

export const useModelingConfiguration = ({
  configState,
  setConfigState,
  catalogFlags,
  upstreamTargetColumn,
  nodeId,
  modelTypeOptions,
}: UseModelingConfigurationArgs): UseModelingConfigurationResult => {
  const {
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isTrainModelDraftNode,
    isHyperparameterTuningNode,
    isClassOversamplingNode,
    isClassUndersamplingNode,
  } = catalogFlags;

  const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;
  const isModelingNode = isTrainModelDraftNode || isHyperparameterTuningNode;

  const featureTargetSplitConfig = useMemo<FeatureTargetSplitConfig | null>(() => {
    if (!isFeatureTargetSplitNode) {
      return null;
    }
    const targetColumn =
      typeof configState?.target_column === 'string' ? configState.target_column.trim() : '';
    const featureColumns = ensureArrayOfString(configState?.feature_columns)
      .map((value) => (typeof value === 'string' ? value.trim() : ''))
      .filter((value) => Boolean(value));
    return {
      targetColumn,
      featureColumns,
    };
  }, [configState?.feature_columns, configState?.target_column, isFeatureTargetSplitNode]);

  const trainTestSplitConfig = useMemo<TrainTestSplitConfig | null>(() => {
    if (!isTrainTestSplitNode) {
      return null;
    }
    const test_size = typeof configState?.test_size === 'number' ? configState.test_size : 0.2;
    const validation_size =
      typeof configState?.validation_size === 'number' ? configState.validation_size : 0.0;
    const random_state =
      typeof configState?.random_state === 'number' ? configState.random_state : 42;
    const shuffle = typeof configState?.shuffle === 'boolean' ? configState.shuffle : true;
    const stratify = typeof configState?.stratify === 'boolean' ? configState.stratify : false;
    const target_column =
      typeof configState?.target_column === 'string' ? configState.target_column.trim() : '';

    return {
      test_size,
      validation_size,
      random_state,
      shuffle,
      stratify,
      target_column,
    };
  }, [
    configState?.random_state,
    configState?.shuffle,
    configState?.stratify,
    configState?.target_column,
    configState?.test_size,
    configState?.validation_size,
    isTrainTestSplitNode,
  ]);

  const resamplingConfig = useMemo<ClassResamplingConfig>(() => {
    if (!isClassResamplingNode) {
      return null;
    }

    const method = typeof configState?.method === 'string' ? configState.method.trim() : '';
    const targetColumn =
      typeof configState?.target_column === 'string' ? configState.target_column.trim() : '';

    const rawSampling = configState?.sampling_strategy;
    let samplingStrategy: number | string | null = null;
    if (typeof rawSampling === 'number' && Number.isFinite(rawSampling)) {
      samplingStrategy = rawSampling;
    } else if (typeof rawSampling === 'string') {
      const trimmed = rawSampling.trim();
      samplingStrategy = trimmed ? trimmed : null;
    } else if (rawSampling === null) {
      samplingStrategy = null;
    }

    const rawRandomState = configState?.random_state;
    let randomState: number | null = null;
    if (typeof rawRandomState === 'number' && Number.isFinite(rawRandomState)) {
      randomState = rawRandomState;
    } else if (typeof rawRandomState === 'string') {
      const parsed = Number(rawRandomState);
      randomState = Number.isFinite(parsed) ? parsed : null;
    }

    let replacement: boolean | null = null;
    if (isClassUndersamplingNode) {
      const rawReplacement = configState?.replacement;
      replacement =
        typeof rawReplacement === 'boolean'
          ? rawReplacement
          : typeof rawReplacement === 'string'
            ? rawReplacement.trim().toLowerCase() === 'true'
            : Boolean(rawReplacement);
    }

    let kNeighbors: number | null = null;
    if (isClassOversamplingNode) {
      const rawK = configState?.k_neighbors;
      if (typeof rawK === 'number' && Number.isFinite(rawK)) {
        kNeighbors = Math.max(1, Math.round(rawK));
      } else if (typeof rawK === 'string') {
        const parsed = Number(rawK);
        if (Number.isFinite(parsed)) {
          kNeighbors = Math.max(1, Math.round(parsed));
        }
      }
    }

    return {
      method,
      targetColumn,
      samplingStrategy,
      randomState,
      replacement,
      kNeighbors,
    };
  }, [
    configState?.k_neighbors,
    configState?.method,
    configState?.random_state,
    configState?.replacement,
    configState?.sampling_strategy,
    configState?.target_column,
    isClassOversamplingNode,
    isClassResamplingNode,
    isClassUndersamplingNode,
  ]);

  const rawProblemType =
    typeof configState?.problem_type === 'string'
      ? configState.problem_type.trim().toLowerCase()
      : '';

  const normalizedProblemType: 'classification' | 'regression' =
    rawProblemType === 'regression' ? 'regression' : 'classification';

  const availableModelTypeOptions = useMemo<FeatureNodeParameterOption[]>(() => {
    if (!Array.isArray(modelTypeOptions)) {
      return [];
    }
    return modelTypeOptions.filter((option) => option && typeof option.value === 'string');
  }, [modelTypeOptions]);

  const modelTypeBuckets = useMemo(() => {
    const buckets: Record<'classification' | 'regression', FeatureNodeParameterOption[]> = {
      classification: [],
      regression: [],
    };

    availableModelTypeOptions.forEach((option) => {
      const metadataType = String(option?.metadata?.problem_type ?? '').trim().toLowerCase();
      if (metadataType === 'regression') {
        buckets.regression.push(option);
      } else if (metadataType === 'classification') {
        buckets.classification.push(option);
      } else {
        buckets.classification.push(option);
        buckets.regression.push(option);
      }
    });

    return buckets;
  }, [availableModelTypeOptions]);

  const filteredModelTypeOptions = useMemo(() => {
    if (!availableModelTypeOptions.length) {
      return [];
    }

    const bucket = normalizedProblemType === 'regression'
      ? modelTypeBuckets.regression
      : modelTypeBuckets.classification;

    if (bucket.length > 0) {
      return bucket;
    }

    return availableModelTypeOptions;
  }, [availableModelTypeOptions, modelTypeBuckets, normalizedProblemType]);

  const trainModelDraftConfig = useMemo<TrainModelDraftConfig | null>(() => {
    if (!isModelingNode) {
      return null;
    }
    const targetColumn =
      typeof configState?.target_column === 'string' ? configState.target_column.trim() : '';
    return {
      targetColumn,
      problemType: normalizedProblemType,
    };
  }, [configState?.target_column, isModelingNode, normalizedProblemType]);

  const trainModelCVConfig = useMemo<TrainModelCVConfig>(() => {
    if (!isModelingNode) {
      return {
        enabled: false,
        strategy: 'auto',
        folds: 5,
        shuffle: true,
        randomState: 42,
        refitStrategy: 'train_plus_validation',
      };
    }

    const enabled = configState?.cv_enabled === undefined
      ? Boolean(isHyperparameterTuningNode)
      : Boolean(configState.cv_enabled);

    const rawStrategy =
      typeof configState?.cv_strategy === 'string'
        ? configState.cv_strategy.trim().toLowerCase()
        : 'auto';
    const strategy: TrainModelCVConfig['strategy'] = (['auto', 'kfold', 'stratified_kfold'] as const).includes(
      rawStrategy as TrainModelCVConfig['strategy'],
    )
      ? (rawStrategy as TrainModelCVConfig['strategy'])
      : 'auto';

    const foldsValue = Number(configState?.cv_folds);
    const folds = Number.isFinite(foldsValue) && foldsValue >= 2 ? Math.floor(foldsValue) : 5;

    const shuffle = configState?.cv_shuffle === undefined ? true : Boolean(configState.cv_shuffle);

    let randomState: number | null = null;
    if (typeof configState?.cv_random_state === 'number' && Number.isFinite(configState.cv_random_state)) {
      randomState = Math.trunc(configState.cv_random_state);
    } else if (typeof configState?.cv_random_state === 'string') {
      const parsed = Number(configState.cv_random_state.trim());
      randomState = Number.isFinite(parsed) ? Math.trunc(parsed) : null;
    }

    const rawRefit =
      typeof configState?.cv_refit_strategy === 'string'
        ? configState.cv_refit_strategy.trim().toLowerCase()
        : 'train_plus_validation';
    const refitStrategy: TrainModelCVConfig['refitStrategy'] =
      rawRefit === 'train_only' ? 'train_only' : 'train_plus_validation';

    return {
      enabled,
      strategy,
      folds,
      shuffle,
      randomState,
      refitStrategy,
    };
  }, [configState?.cv_enabled, configState?.cv_folds, configState?.cv_random_state, configState?.cv_refit_strategy, configState?.cv_shuffle, configState?.cv_strategy, isHyperparameterTuningNode, isModelingNode]);

  useEffect(() => {
    if (!isModelingNode) {
      return;
    }

    if (rawProblemType === normalizedProblemType) {
      return;
    }

    setConfigState((previous) => ({
      ...previous,
      problem_type: normalizedProblemType,
    }));
  }, [
    isModelingNode,
    normalizedProblemType,
    rawProblemType,
    setConfigState,
  ]);

  useEffect(() => {
    if (!isModelingNode) {
      return;
    }

    if (!filteredModelTypeOptions.length) {
      return;
    }

    const allowedValues = new Set(filteredModelTypeOptions.map((option) => option.value));
    const currentModelType =
      typeof configState?.model_type === 'string' ? configState.model_type.trim() : '';

    if (currentModelType && allowedValues.has(currentModelType)) {
      return;
    }

    const fallbackModelType = filteredModelTypeOptions[0]?.value ?? '';
    if (!fallbackModelType) {
      return;
    }

    setConfigState((previous) => {
      const previousValue =
        typeof previous?.model_type === 'string' ? previous.model_type.trim() : '';
      if (previousValue && allowedValues.has(previousValue)) {
        return previous;
      }
      if (previousValue === fallbackModelType) {
        return previous;
      }

      return {
        ...previous,
        model_type: fallbackModelType,
      };
    });
  }, [
    configState?.model_type,
    filteredModelTypeOptions,
    isModelingNode,
    setConfigState,
  ]);

  const trainModelRuntimeConfig = useMemo<TrainModelRuntimeConfig | null>(() => {
    if (!isModelingNode) {
      return null;
    }

    const rawModelType =
      typeof configState?.model_type === 'string' ? configState.model_type.trim() : '';

    const rawHyperparameters = configState?.hyperparameters;
    let hyperparameters: Record<string, any> | null = null;
    let hyperparametersError: string | null = null;

    if (rawHyperparameters && typeof rawHyperparameters === 'string') {
      const trimmed = rawHyperparameters.trim();
      if (trimmed.length) {
        try {
          const parsed = JSON.parse(trimmed);
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            hyperparameters = parsed as Record<string, any>;
          } else {
            hyperparametersError = 'Hyperparameters JSON must be an object.';
          }
        } catch (error) {
          hyperparametersError = 'Hyperparameters JSON is invalid. Provide a valid JSON object.';
        }
      }
    } else if (
      rawHyperparameters &&
      typeof rawHyperparameters === 'object' &&
      !Array.isArray(rawHyperparameters)
    ) {
      hyperparameters = rawHyperparameters as Record<string, any>;
    }

    return {
      modelType: rawModelType || null,
      hyperparameters,
      hyperparametersError,
    };
  }, [configState?.hyperparameters, configState?.model_type, isModelingNode]);

  useEffect(() => {
    if (!(isTrainTestSplitNode || isTrainModelDraftNode || isHyperparameterTuningNode)) {
      return;
    }

    if (!upstreamTargetColumn) {
      return;
    }

    if (isTrainTestSplitNode && configState?.stratify !== true) {
      return;
    }

    setConfigState((previous) => {
      const existingTarget =
        typeof previous?.target_column === 'string' ? previous.target_column.trim() : '';
      if (existingTarget) {
        return previous;
      }

      return {
        ...previous,
        target_column: upstreamTargetColumn,
      };
    });
  }, [
    configState?.stratify,
    isHyperparameterTuningNode,
    isTrainModelDraftNode,
    isTrainTestSplitNode,
    nodeId,
    setConfigState,
    upstreamTargetColumn,
  ]);

  return {
    featureTargetSplitConfig,
    trainTestSplitConfig,
    resamplingConfig,
    trainModelDraftConfig,
    trainModelRuntimeConfig,
    trainModelCVConfig,
    filteredModelTypeOptions,
  };
};
