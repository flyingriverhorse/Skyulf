import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type {
  FeatureGraph,
  FeatureNodeParameter,
  FeatureNodeParameterOption,
  HyperparameterTuningJobCreatePayload,
  HyperparameterTuningJobListResponse,
  HyperparameterTuningJobResponse,
  HyperparameterTuningJobSummary,
  ModelHyperparameterField,
  ModelHyperparametersResponse,
} from '../../../../api';
import {
  createHyperparameterTuningJob,
  fetchHyperparameterTuningJobs,
  cancelHyperparameterTuningJob,
  fetchModelHyperparameters,
  generatePipelineId,
  type FetchHyperparameterTuningJobsOptions,
} from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';
import { stableStringify } from '../../utils/configParsers';
import type { TrainModelDraftConfig } from './TrainModelDraftSection';
import type { TrainModelCVConfig } from '../../hooks';
import { useScalingWarning, detectScalingConvergenceFromJob, hasScalingConvergenceMessage } from './useScalingWarning';

const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const TUNING_STRATEGIES_KEY = '__tuning_strategies';
const TUNING_ACTIVE_ID_KEY = '__active_tuning_strategy_id';
const TUNING_EXPANDED_KEY = '__expanded_tuning_strategy_ids';

const SEARCH_STRATEGY_FALLBACKS = ['random', 'grid', 'halving', 'optuna'];

type SearchStrategy = string;

const normalizeSearchStrategyValue = (
  value: unknown,
  knownValues?: readonly string[],
): SearchStrategy | null => {
  if (typeof value !== 'string') {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed.length) {
    return null;
  }
  const lowered = trimmed.toLowerCase();

  if (knownValues && knownValues.length > 0) {
    const match = knownValues.find((candidate) => candidate.toLowerCase() === lowered);
    if (match) {
      return match;
    }
  }

  const fallbackMatch = SEARCH_STRATEGY_FALLBACKS.find((candidate) => candidate.toLowerCase() === lowered);
  if (fallbackMatch) {
    return fallbackMatch;
  }

  return trimmed;
};

type HyperparameterTuningSectionProps = {
  nodeId: string;
  sourceId?: string | null;
  graph: FeatureGraph | null;
  config: TrainModelDraftConfig | null;
  runtimeConfig: { modelType: string | null } | null;
  cvConfig: TrainModelCVConfig;
  draftConfigState: Record<string, any> | null;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  modelTypeParameter: FeatureNodeParameter | null;
  modelTypeOptions: FeatureNodeParameterOption[];
  searchStrategyParameter: FeatureNodeParameter | null;
  searchIterationsParameter: FeatureNodeParameter | null;
  searchRandomStateParameter: FeatureNodeParameter | null;
  scoringMetricParameter: FeatureNodeParameter | null;
  setDraftConfigState: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  onSaveDraftConfig?: (options?: { closeModal?: boolean }) => void | Promise<void>;
  onUpdateNodeData?: (nodeId: string, dataUpdates: Record<string, any>) => void;
  currentStatus?: string;
  currentProgress?: number;
};

type ParsedJsonResult<T> = {
  value: T | null;
  error: string | null;
};

const isRecord = (value: unknown): value is Record<string, any> => {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
};

const cloneJson = <T,>(value: T): T => {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch (error) {
    return value;
  }
};

const parseJsonObject = (raw: unknown): ParsedJsonResult<Record<string, any>> => {
  if (raw === null || raw === undefined) {
    return { value: null, error: null };
  }
  if (typeof raw === 'string') {
    const trimmed = raw.trim();
    if (!trimmed.length) {
      return { value: null, error: null };
    }
    try {
      const parsed = JSON.parse(trimmed);
      if (isRecord(parsed)) {
        return { value: parsed, error: null };
      }
      return { value: null, error: 'JSON must evaluate to an object.' };
    } catch (error) {
      return { value: null, error: 'Invalid JSON. Provide a valid JSON object.' };
    }
  }
  if (isRecord(raw)) {
    return { value: raw, error: null };
  }
  return { value: null, error: 'Provide a JSON object for this field.' };
};

const normalizeSearchSpace = (
  raw: Record<string, any> | null,
): ParsedJsonResult<Record<string, any[]>> => {
  if (!raw) {
    return { value: null, error: null };
  }

  const sanitized: Record<string, any[]> = {};
  Object.entries(raw).forEach(([key, value]) => {
    if (!key) {
      return;
    }
    if (Array.isArray(value)) {
      const cleaned = value.filter((entry) => entry !== undefined && entry !== null);
      if (cleaned.length > 0) {
        sanitized[key] = cleaned.map((entry) => cloneJson(entry));
      }
      return;
    }
    if (value === null || value === undefined || value === '') {
      return;
    }
    sanitized[key] = [cloneJson(value)];
  });

  if (!Object.keys(raw).length) {
    return { value: null, error: 'Populate the search space with at least one hyperparameter.' };
  }

  if (Object.keys(sanitized).length === 0) {
    return {
      value: null,
      error: 'Search space must list candidate values for at least one hyperparameter.',
    };
  }

  return { value: sanitized, error: null };
};

const formatMetricValue = (value: number): string => {
  if (!Number.isFinite(value)) {
    return '';
  }
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 100) {
    return value.toFixed(1);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
};

const KNOWN_SPLITS = new Set(['train', 'test', 'validation']);

const sanitizeSplitList = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return [];
  }

  const unique = new Set<string>();
  value.forEach((entry) => {
    if (typeof entry === 'string') {
      const lowered = entry.trim().toLowerCase();
      if (KNOWN_SPLITS.has(lowered)) {
        unique.add(lowered);
      }
    }
  });

  return Array.from(unique);
};

const filterHyperparametersByFields = (
  values: Record<string, any> | null | undefined,
  fieldNames: Set<string>,
): Record<string, any> => {
  if (!values || fieldNames.size === 0) {
    return {};
  }
  const filtered: Record<string, any> = {};
  Object.entries(values).forEach(([key, value]) => {
    if (fieldNames.has(key)) {
      filtered[key] = value;
    }
  });
  return filtered;
};

const filterSearchSpaceByFields = (
  values: Record<string, any> | null | undefined,
  fieldNames: Set<string>,
): Record<string, any[]> => {
  if (!values || fieldNames.size === 0) {
    return {};
  }
  const filtered: Record<string, any[]> = {};
  Object.entries(values).forEach(([key, rawValue]) => {
    if (!fieldNames.has(key)) {
      return;
    }
    if (Array.isArray(rawValue)) {
      const cleaned = rawValue
        .map((entry) => cloneJson(entry))
        .filter((entry) => entry !== undefined && entry !== null && entry !== '');
      if (cleaned.length > 0) {
        filtered[key] = cleaned;
      }
      return;
    }
    if (rawValue !== undefined && rawValue !== null && rawValue !== '') {
      filtered[key] = [cloneJson(rawValue)];
    }
  });
  return filtered;
};

type SearchSpaceParseResult = {
  values: any[];
  error: string | null;
};

const parseSearchInput = (
  field: ModelHyperparameterField,
  rawText: string,
): SearchSpaceParseResult => {
  const tokens = rawText
    .split(/[\n,]+/)
    .map((token) => token.trim())
    .filter((token) => token.length > 0);

  if (tokens.length === 0) {
    return { values: [], error: null };
  }

  const values: any[] = [];
  const invalidTokens: string[] = [];
  const selectOptions = Array.isArray(field.options)
    ? new Set(field.options.map((option) => option.value))
    : null;

  tokens.forEach((token) => {
    if (field.type === 'number') {
      const parsed = Number(token);
      if (Number.isFinite(parsed)) {
        values.push(parsed);
      } else {
        invalidTokens.push(token);
      }
      return;
    }

    if (field.type === 'boolean') {
      const lowered = token.toLowerCase();
      if (['true', '1', 'yes', 'y'].includes(lowered)) {
        values.push(true);
        return;
      }
      if (['false', '0', 'no', 'n'].includes(lowered)) {
        values.push(false);
        return;
      }
      invalidTokens.push(token);
      return;
    }

    if (field.type === 'select') {
      if (selectOptions && !selectOptions.has(token)) {
        invalidTokens.push(token);
        return;
      }
      values.push(token);
      return;
    }

    values.push(token);
  });

  const dedupedValues: any[] = [];
  const seenKeys = new Set<string>();
  values.forEach((value) => {
    const key = typeof value === 'object' ? JSON.stringify(value) : String(value);
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      dedupedValues.push(value);
    }
  });

  if (invalidTokens.length > 0) {
    const preview = invalidTokens.slice(0, 3).join(', ');
    const suffix = invalidTokens.length > 3 ? '…' : '';
    return {
      values: dedupedValues,
      error: `Invalid value${invalidTokens.length > 1 ? 's' : ''}: ${preview}${suffix}`,
    };
  }

  return { values: dedupedValues, error: null };
};

const formatSearchValue = (field: ModelHyperparameterField, value: any): string => {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toString() : '';
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  if (field.type === 'select' && typeof value === 'string') {
    return value;
  }
  return String(value);
};

const formatValuePreview = (value: any): string => {
  if (value === null || value === undefined) {
    return '—';
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toString() : '—';
  }
  if (typeof value === 'boolean') {
    return value ? 'True' : 'False';
  }
  if (typeof value === 'string') {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch (error) {
    return String(value);
  }
};

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim().length) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const clampNumber = (value: number, min?: number | null, max?: number | null): number => {
  let result = value;
  if (typeof min === 'number' && Number.isFinite(min)) {
    result = Math.max(result, min);
  }
  if (typeof max === 'number' && Number.isFinite(max)) {
    result = Math.min(result, max);
  }
  return result;
};

const buildDefaultSearchSpace = (
  fields: ModelHyperparameterField[],
  defaults: Record<string, any>,
): Record<string, any[]> => {
  const space: Record<string, any[]> = {};

  const solverField = fields.find((field) => field?.name === 'solver');
  const penaltyField = fields.find((field) => field?.name === 'penalty');
  const isLogisticRegressionConfig = Boolean(
    solverField &&
      penaltyField &&
      Array.isArray(solverField.options) &&
      solverField.options.some((option) => option?.value === 'liblinear') &&
      Array.isArray(penaltyField.options) &&
      penaltyField.options.some((option) => option?.value === 'elasticnet'),
  );
  const logisticSafeSolvers = new Set(['lbfgs', 'saga']);
  const logisticSafePenalties = new Set(['l2', 'none']);

  fields.forEach((field) => {
    if (!field?.name) {
      return;
    }

    const candidateMap = new Map<string, any>();
    const addCandidate = (value: any) => {
      if (value === undefined || value === null) {
        return;
      }
      if (typeof value === 'number' && !Number.isFinite(value)) {
        return;
      }
      const key = typeof value === 'object' ? JSON.stringify(value) : `${typeof value}:${String(value)}`;
      if (!candidateMap.has(key)) {
        candidateMap.set(key, cloneJson(value));
      }
    };

    const defaultValue = defaults?.[field.name];

    if (isLogisticRegressionConfig && field.name === 'solver') {
      const options = Array.isArray(field.options) ? field.options : [];
      options.forEach((option) => {
        if (option?.value && logisticSafeSolvers.has(String(option.value))) {
          addCandidate(option.value);
        }
      });
      if (!candidateMap.size && typeof defaultValue === 'string' && logisticSafeSolvers.has(defaultValue)) {
        addCandidate(defaultValue);
      }
      if (!candidateMap.size) {
        logisticSafeSolvers.forEach((solver) => addCandidate(solver));
      }
    } else if (isLogisticRegressionConfig && field.name === 'penalty') {
      const options = Array.isArray(field.options) ? field.options : [];
      options.forEach((option) => {
        if (option?.value && logisticSafePenalties.has(String(option.value))) {
          addCandidate(option.value);
        }
      });
      if (!candidateMap.size && typeof defaultValue === 'string' && logisticSafePenalties.has(defaultValue)) {
        addCandidate(defaultValue);
      }
      if (!candidateMap.size) {
        logisticSafePenalties.forEach((penalty) => addCandidate(penalty));
      }
    } else if (field.type === 'number') {
      const numericDefault = toFiniteNumber(defaultValue);
      const numericMin = toFiniteNumber(field.min);
      const numericMax = toFiniteNumber(field.max);

      if (numericMin !== null) {
        addCandidate(numericMin);
      }
      if (numericDefault !== null) {
        addCandidate(clampNumber(numericDefault, numericMin, numericMax));
      }
      if (numericMax !== null) {
        addCandidate(numericMax);
      }

      if (candidateMap.size <= 1 && numericDefault !== null) {
        const step = toFiniteNumber(field.step);
        if (step !== null && step > 0) {
          addCandidate(clampNumber(numericDefault + step, numericMin, numericMax));
          addCandidate(clampNumber(numericDefault - step, numericMin, numericMax));
        } else {
          addCandidate(clampNumber(numericDefault * 1.5, numericMin, numericMax));
          addCandidate(clampNumber(numericDefault * 0.5, numericMin, numericMax));
        }
      }
    } else if (field.type === 'boolean') {
      addCandidate(true);
      addCandidate(false);
      if (typeof defaultValue === 'boolean') {
        addCandidate(defaultValue);
      }
    } else if (field.type === 'select') {
      (field.options ?? []).slice(0, 6).forEach((option) => {
        if (option?.value !== undefined) {
          addCandidate(option.value);
        }
      });
      if (typeof defaultValue === 'string' && defaultValue.trim().length) {
        addCandidate(defaultValue.trim());
      }
    } else {
      if (typeof defaultValue === 'string' && defaultValue.trim().length) {
        addCandidate(defaultValue.trim());
      }
    }

    const collected = Array.from(candidateMap.values());
    if (collected.length > 0) {
      space[field.name] = collected;
    }
  });

  return space;
};

const SCORING_SUGGESTIONS: Record<'classification' | 'regression', Array<{ value: string; label: string }>> = {
  classification: [
    { value: 'accuracy', label: 'Accuracy' },
    { value: 'balanced_accuracy', label: 'Balanced accuracy' },
    { value: 'f1', label: 'F1' },
    { value: 'f1_macro', label: 'F1 macro' },
    { value: 'f1_weighted', label: 'F1 weighted' },
    { value: 'precision', label: 'Precision' },
    { value: 'recall', label: 'Recall' },
    { value: 'roc_auc', label: 'ROC AUC (binary)' },
    { value: 'roc_auc_ovr', label: 'ROC AUC (one-vs-rest)' },
  ],
  regression: [
    { value: 'neg_mean_squared_error', label: 'Negative MSE' },
    { value: 'neg_root_mean_squared_error', label: 'Negative RMSE' },
    { value: 'neg_mean_absolute_error', label: 'Negative MAE' },
    { value: 'neg_median_absolute_error', label: 'Negative MedAE' },
    { value: 'r2', label: 'R²' },
  ],
};

type TuningStrategyDraft = {
  id: string;
  label: string;
  config: Record<string, any>;
  searchSpaceTexts: Record<string, string>;
  searchSpaceFieldErrors: Record<string, string | null>;
  hasAutoFilledSearchSpace: boolean;
  searchSpaceManuallyCleared: boolean;
};

const createStrategyId = () => `strategy-${Math.random().toString(36).slice(2, 10)}`;

const stripTuningMetadata = (config: Record<string, any> | null | undefined): Record<string, any> => {
  if (!isRecord(config)) {
    return {};
  }
  const cloned = cloneJson(config);
  if (!isRecord(cloned)) {
    return {};
  }
  delete cloned[TUNING_STRATEGIES_KEY];
  delete cloned[TUNING_ACTIVE_ID_KEY];
  delete cloned[TUNING_EXPANDED_KEY];
  return cloned;
};

const cloneConfigForStrategy = (config: Record<string, any> | null | undefined): Record<string, any> => {
  return stripTuningMetadata(config);
};

const createStrategyDraft = (
  label: string,
  config: Record<string, any> | null | undefined,
  extras?: Partial<Omit<TuningStrategyDraft, 'id' | 'label' | 'config'>>,
): TuningStrategyDraft => ({
  id: createStrategyId(),
  label,
  config: cloneConfigForStrategy(config),
  searchSpaceTexts: extras?.searchSpaceTexts ? { ...extras.searchSpaceTexts } : {},
  searchSpaceFieldErrors: extras?.searchSpaceFieldErrors ? { ...extras.searchSpaceFieldErrors } : {},
  hasAutoFilledSearchSpace: extras?.hasAutoFilledSearchSpace ?? false,
  searchSpaceManuallyCleared: extras?.searchSpaceManuallyCleared ?? false,
});

const createStrategyLabel = (ordinal: number): string => `Strategy ${ordinal}`;

type PersistedStrategyEntry = {
  id: string;
  label: string;
  config: Record<string, any>;
  searchSpaceTexts: Record<string, string>;
  searchSpaceFieldErrors: Record<string, string | null>;
  hasAutoFilledSearchSpace: boolean;
  searchSpaceManuallyCleared: boolean;
};

type PersistedStrategyMetadata = {
  strategies: PersistedStrategyEntry[];
  activeId: string | null;
  expandedIds: string[];
};

const sanitizeStringMap = (raw: unknown): Record<string, string> => {
  if (!isRecord(raw)) {
    return {};
  }
  return Object.keys(raw)
    .filter((key) => typeof raw[key] === 'string')
    .sort()
    .reduce<Record<string, string>>((accumulator, key) => {
      accumulator[key] = String(raw[key]);
      return accumulator;
    }, {});
};

const sanitizeNullableStringMap = (raw: unknown): Record<string, string | null> => {
  if (!isRecord(raw)) {
    return {};
  }
  return Object.keys(raw)
    .filter((key) => raw[key] === null || typeof raw[key] === 'string')
    .sort()
    .reduce<Record<string, string | null>>((accumulator, key) => {
      const value = raw[key];
      accumulator[key] = value === null ? null : String(value);
      return accumulator;
    }, {});
};

const extractStoredStrategyMetadataFields = (config: Record<string, any> | null | undefined): Record<string, any> => {
  if (!isRecord(config)) {
    return {};
  }
    const metadata: Record<string, any> = {};
  if (Object.prototype.hasOwnProperty.call(config, TUNING_STRATEGIES_KEY)) {
    metadata[TUNING_STRATEGIES_KEY] = cloneJson(config[TUNING_STRATEGIES_KEY]);
  }
  if (Object.prototype.hasOwnProperty.call(config, TUNING_ACTIVE_ID_KEY)) {
    metadata[TUNING_ACTIVE_ID_KEY] = config[TUNING_ACTIVE_ID_KEY];
  }
  if (Object.prototype.hasOwnProperty.call(config, TUNING_EXPANDED_KEY)) {
    metadata[TUNING_EXPANDED_KEY] = cloneJson(config[TUNING_EXPANDED_KEY]);
  }
  return metadata;
};

const createEmptyPersistedMetadata = (): PersistedStrategyMetadata => ({
  strategies: [],
  activeId: null,
  expandedIds: [],
});

const convertConfigToPersistedMetadata = (
  config: Record<string, any> | null | undefined,
): PersistedStrategyMetadata => {
  const metadata = createEmptyPersistedMetadata();
  if (!isRecord(config)) {
    return metadata;
  }

  const rawStrategies = config[TUNING_STRATEGIES_KEY];
  if (Array.isArray(rawStrategies)) {
    const seenIds = new Set<string>();
    rawStrategies.forEach((entry) => {
      if (!isRecord(entry)) {
        return;
      }
      let id = typeof entry.id === 'string' ? entry.id.trim() : '';
      if (!id || seenIds.has(id)) {
        id = createStrategyId();
      }
      seenIds.add(id);

      const labelRaw = typeof entry.label === 'string' ? entry.label.trim() : '';
      const label = labelRaw.length ? labelRaw : createStrategyLabel(metadata.strategies.length + 1);

      const baseConfig = isRecord(entry.config) ? entry.config : {};
      const strategyEntry: PersistedStrategyEntry = {
        id,
        label,
        config: stripTuningMetadata(baseConfig),
        searchSpaceTexts: sanitizeStringMap(entry.searchSpaceTexts),
        searchSpaceFieldErrors: sanitizeNullableStringMap(entry.searchSpaceFieldErrors),
        hasAutoFilledSearchSpace: Boolean(entry.hasAutoFilledSearchSpace),
        searchSpaceManuallyCleared: Boolean(entry.searchSpaceManuallyCleared),
      };
      metadata.strategies.push(strategyEntry);
    });
  }

  const availableIds = new Set(metadata.strategies.map((entry) => entry.id));

  const rawActiveId = config[TUNING_ACTIVE_ID_KEY];
  const candidateActiveId = typeof rawActiveId === 'string' ? rawActiveId.trim() : '';
  metadata.activeId = candidateActiveId && availableIds.has(candidateActiveId)
    ? candidateActiveId
    : metadata.strategies[0]?.id ?? null;

  const rawExpanded = config[TUNING_EXPANDED_KEY];
  const expandedIds = Array.isArray(rawExpanded)
    ? rawExpanded
        .map((entry) => (typeof entry === 'string' ? entry.trim() : ''))
        .filter((entry): entry is string => Boolean(entry) && availableIds.has(entry))
    : [];

  if (metadata.activeId && !expandedIds.includes(metadata.activeId)) {
    expandedIds.push(metadata.activeId);
  }

  metadata.expandedIds = Array.from(new Set(expandedIds)).sort();

  return metadata;
};

const buildStrategyDraftFromPersisted = (entry: PersistedStrategyEntry): TuningStrategyDraft => ({
  id: entry.id,
  label: entry.label,
  config: cloneConfigForStrategy(entry.config),
  searchSpaceTexts: { ...entry.searchSpaceTexts },
  searchSpaceFieldErrors: { ...entry.searchSpaceFieldErrors },
  hasAutoFilledSearchSpace: entry.hasAutoFilledSearchSpace,
  searchSpaceManuallyCleared: entry.searchSpaceManuallyCleared,
});

const convertDraftStateToPersistedMetadata = (
  strategies: TuningStrategyDraft[],
  activeStrategyId: string,
  expandedStrategyIds: Set<string>,
): PersistedStrategyMetadata => {
  const persistedStrategies = strategies.map((strategy, index) => ({
    id: typeof strategy.id === 'string' && strategy.id.trim().length ? strategy.id : createStrategyId(),
    label:
      typeof strategy.label === 'string' && strategy.label.trim().length
        ? strategy.label
        : createStrategyLabel(index + 1),
    config: stripTuningMetadata(strategy.config),
    searchSpaceTexts: sanitizeStringMap(strategy.searchSpaceTexts),
    searchSpaceFieldErrors: sanitizeNullableStringMap(strategy.searchSpaceFieldErrors),
    hasAutoFilledSearchSpace: Boolean(strategy.hasAutoFilledSearchSpace),
    searchSpaceManuallyCleared: Boolean(strategy.searchSpaceManuallyCleared),
  }));

  const availableIds = new Set(persistedStrategies.map((entry) => entry.id));
  const resolvedActiveId =
    typeof activeStrategyId === 'string' && availableIds.has(activeStrategyId)
      ? activeStrategyId
      : persistedStrategies[0]?.id ?? null;

  const expandedArray = Array.from(expandedStrategyIds ?? new Set<string>())
    .filter((id) => availableIds.has(id));
  if (resolvedActiveId && !expandedArray.includes(resolvedActiveId)) {
    expandedArray.push(resolvedActiveId);
  }

  return {
    strategies: persistedStrategies,
    activeId: resolvedActiveId,
    expandedIds: Array.from(new Set(expandedArray)).sort(),
  };
};

const resolveActiveStrategyId = (
  strategies: TuningStrategyDraft[],
  requestedId: string | null,
): string => {
  if (requestedId && strategies.some((strategy) => strategy.id === requestedId)) {
    return requestedId;
  }
  return strategies[0]?.id ?? '';
};

const resolveExpandedStrategyIds = (
  strategies: TuningStrategyDraft[],
  requestedIds: string[],
): Set<string> => {
  const available = new Set(strategies.map((strategy) => strategy.id));
  const resolved = requestedIds.filter((id) => available.has(id));
  if (!resolved.length && strategies[0]) {
    resolved.push(strategies[0].id);
  }
  return new Set(resolved);
};

const buildMetadataSignature = (metadata: PersistedStrategyMetadata): string => {
  return stableStringify(metadata);
};

const shallowEqualRecord = (a: Record<string, any>, b: Record<string, any>): boolean => {
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) {
    return false;
  }
  for (const key of keysA) {
    if (a[key] !== b[key]) {
      return false;
    }
  }
  return true;
};

export const HyperparameterTuningSection: React.FC<HyperparameterTuningSectionProps> = ({
  nodeId,
  sourceId,
  graph,
  config,
  runtimeConfig,
  cvConfig,
  draftConfigState,
  renderParameterField,
  modelTypeParameter,
  modelTypeOptions,
  searchStrategyParameter,
  searchIterationsParameter,
  searchRandomStateParameter,
  scoringMetricParameter,
  setDraftConfigState,
  onSaveDraftConfig,
  onUpdateNodeData,
  currentStatus,
  currentProgress,
}) => {
  const queryClient = useQueryClient();

  const cancelMutation = useMutation({
    mutationFn: cancelHyperparameterTuningJob,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['hyperparameter-tuning-jobs'] });
    },
    onError: (error) => {
      console.error('Failed to cancel job:', error);
      alert(error instanceof Error ? error.message : 'Failed to cancel job');
    },
  });

  const handleCancel = (jobId: string) => {
    if (confirm('Are you sure you want to cancel this tuning job?')) {
      cancelMutation.mutate(jobId);
    }
  };

  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [lastCreatedJob, setLastCreatedJob] = useState<HyperparameterTuningJobResponse | null>(null);
  const [lastCreatedJobCount, setLastCreatedJobCount] = useState<number>(0);
  const [pipelineIdFromSavedConfig, setPipelineIdFromSavedConfig] = useState<string | null>(null);
  const [hasDraftChanges, setHasDraftChanges] = useState(false);
  const [batchEnqueueError, setBatchEnqueueError] = useState<string | null>(null);
  const [showScalingDetails, setShowScalingDetails] = useState(false);

  const targetColumn = (config?.targetColumn ?? '').trim();
  const problemType = config?.problemType === 'regression' ? 'regression' : 'classification';
  const modelTypeParameterName = modelTypeParameter?.name ?? 'model_type';
  const searchStrategyParameterName = searchStrategyParameter?.name ?? 'search_strategy';
  const searchIterationsParameterName = searchIterationsParameter?.name ?? 'search_iterations';
  const searchRandomStateParameterName = searchRandomStateParameter?.name ?? 'search_random_state';
  const scoringMetricParameterName = scoringMetricParameter?.name ?? 'scoring_metric';
  const selectedModelTypeRaw = draftConfigState?.[modelTypeParameterName];
  const modelType = typeof selectedModelTypeRaw === 'string' && selectedModelTypeRaw.trim().length
    ? selectedModelTypeRaw.trim()
    : (runtimeConfig?.modelType ?? '').trim();

  const hyperparamQuery = useQuery<ModelHyperparametersResponse, Error>({
    queryKey: ['model-hyperparameters', modelType],
    queryFn: () => fetchModelHyperparameters(modelType),
    enabled: Boolean(modelType),
    staleTime: 5 * 60 * 1000,
  });

  const hyperparamFields = hyperparamQuery.data?.fields ?? [];
  const hyperparamDefaults = hyperparamQuery.data?.defaults ?? {};

  const allowedHyperparamNames = useMemo(() => {
    if (!hyperparamFields.length) {
      return new Set<string>();
    }
    const fieldNames = new Set<string>();
    hyperparamFields.forEach((field) => {
      if (field?.name) {
        fieldNames.add(field.name);
      }
    });
    return fieldNames;
  }, [hyperparamFields]);

  const cvEnabled = Boolean(cvConfig?.enabled);
  const cvStrategy = cvConfig?.strategy ?? 'auto';
  const cvFolds = typeof cvConfig?.folds === 'number' ? cvConfig.folds : 5;
  const cvShuffle = Boolean(cvConfig?.shuffle);
  const cvRandomState = cvConfig?.randomState ?? null;

  const searchStrategyOptions = useMemo<string[]>(() => {
    const values: string[] = [];
    (searchStrategyParameter?.options ?? []).forEach((option) => {
      if (!option || typeof option.value !== 'string') {
        return;
      }
      const trimmed = option.value.trim();
      if (!trimmed.length) {
        return;
      }
      const lowered = trimmed.toLowerCase();
      if (!values.some((candidate) => candidate.toLowerCase() === lowered)) {
        values.push(trimmed);
      }
    });
    return values;
  }, [searchStrategyParameter]);

  const availableSearchStrategies = useMemo<SearchStrategy[]>(() => {
    if (searchStrategyOptions.length > 0) {
      return [...searchStrategyOptions];
    }
    return [...SEARCH_STRATEGY_FALLBACKS];
  }, [searchStrategyOptions]);

  const parameterDefaultStrategy = normalizeSearchStrategyValue(
    searchStrategyParameter?.default,
    searchStrategyOptions,
  );
  const draftSearchStrategy = normalizeSearchStrategyValue(
    draftConfigState?.[searchStrategyParameterName],
    searchStrategyOptions,
  );
  const searchStrategy: SearchStrategy =
    draftSearchStrategy ??
    parameterDefaultStrategy ??
    availableSearchStrategies[0] ??
    SEARCH_STRATEGY_FALLBACKS[0];

  const rawIterations = draftConfigState?.[searchIterationsParameterName];
  let searchIterations: number | null = null;
  if (typeof rawIterations === 'number' && Number.isFinite(rawIterations)) {
    searchIterations = Math.max(1, Math.floor(rawIterations));
  } else if (typeof rawIterations === 'string') {
    const parsed = Number(rawIterations.trim());
    if (Number.isFinite(parsed) && parsed > 0) {
      searchIterations = Math.max(1, Math.floor(parsed));
    }
  }

  const rawSearchRandomState = draftConfigState?.[searchRandomStateParameterName];
  let searchRandomState: number | null = null;
  if (typeof rawSearchRandomState === 'number' && Number.isFinite(rawSearchRandomState)) {
    searchRandomState = Math.trunc(rawSearchRandomState);
  } else if (typeof rawSearchRandomState === 'string') {
    const parsed = Number(rawSearchRandomState.trim());
    if (Number.isFinite(parsed)) {
      searchRandomState = Math.trunc(parsed);
    }
  }

  const searchIterationsDefault =
    typeof searchIterationsParameter?.default === 'number' && Number.isFinite(searchIterationsParameter.default)
      ? Math.max(1, Math.floor(searchIterationsParameter.default))
      : null;

  const searchRandomStateDefault =
    typeof searchRandomStateParameter?.default === 'number' && Number.isFinite(searchRandomStateParameter.default)
      ? Math.trunc(searchRandomStateParameter.default)
      : null;

  const rawScoringMetric = typeof draftConfigState?.[scoringMetricParameterName] === 'string'
    ? String(draftConfigState[scoringMetricParameterName]).trim()
    : '';
  const scoringMetric = rawScoringMetric.length ? rawScoringMetric : null;

  const baselineParse = useMemo(() => parseJsonObject(draftConfigState?.baseline_hyperparameters), [
    draftConfigState?.baseline_hyperparameters,
  ]);

  const searchSpaceParse = useMemo(() => parseJsonObject(draftConfigState?.search_space), [
    draftConfigState?.search_space,
  ]);

  const normalizedSearchSpace = useMemo(
    () => normalizeSearchSpace(searchSpaceParse.value),
    [searchSpaceParse.value],
  );

  const baselineHyperparameters = baselineParse.value;
  const baselineError = baselineParse.error;
  const sanitizedSearchSpace = normalizedSearchSpace.value;
  const searchSpaceError = searchSpaceParse.error ?? normalizedSearchSpace.error;

  const filteredBaselineOverrides = useMemo(
    () => filterHyperparametersByFields(baselineHyperparameters, allowedHyperparamNames),
    [allowedHyperparamNames, baselineHyperparameters],
  );

  const filteredSearchOverrides = useMemo(
    () => filterSearchSpaceByFields(sanitizedSearchSpace, allowedHyperparamNames),
    [allowedHyperparamNames, sanitizedSearchSpace],
  );

  const baselineOverrideCount = useMemo(() => Object.keys(filteredBaselineOverrides).length, [filteredBaselineOverrides]);
  const searchOverrideCount = useMemo(() => Object.keys(filteredSearchOverrides).length, [filteredSearchOverrides]);
  const hasSearchSpaceEntries = searchOverrideCount > 0;

  const [searchSpaceTexts, setSearchSpaceTexts] = useState<Record<string, string>>({});
  const [searchSpaceFieldErrors, setSearchSpaceFieldErrors] = useState<Record<string, string | null>>({});
  const [hasAutoFilledSearchSpace, setHasAutoFilledSearchSpace] = useState(false);
  const [searchSpaceManuallyCleared, setSearchSpaceManuallyCleared] = useState(false);
  const initialMetadataRef = useRef<{
    strategies: TuningStrategyDraft[];
    activeId: string;
    expanded: Set<string>;
    signature: string;
  } | null>(null);

  if (!initialMetadataRef.current) {
    const persistedMetadata = convertConfigToPersistedMetadata(draftConfigState);
    const hasPersistedStrategies = persistedMetadata.strategies.length > 0;

    const buildDefaultStrategyDrafts = (): TuningStrategyDraft[] => {
      // Begin with no predefined strategies so users can choose exactly what they need.
      return [];
    };

    let initialStrategies = hasPersistedStrategies
      ? persistedMetadata.strategies.map(buildStrategyDraftFromPersisted)
      : buildDefaultStrategyDrafts();

    const resolvedActive = resolveActiveStrategyId(initialStrategies, persistedMetadata.activeId);
    const resolvedExpanded = resolveExpandedStrategyIds(initialStrategies, persistedMetadata.expandedIds);
    if (resolvedActive && !resolvedExpanded.has(resolvedActive) && initialStrategies.length) {
      resolvedExpanded.add(resolvedActive);
    }

    const initialSignature = buildMetadataSignature(persistedMetadata);

    initialMetadataRef.current = {
      strategies: initialStrategies,
      activeId: resolvedActive,
      expanded: resolvedExpanded,
      signature: initialSignature,
    };
  }

  const [strategies, setStrategies] = useState<TuningStrategyDraft[]>(() =>
    initialMetadataRef.current ? initialMetadataRef.current.strategies : [],
  );
  const [expandedStrategyIds, setExpandedStrategyIds] = useState<Set<string>>(
    () => new Set(initialMetadataRef.current ? Array.from(initialMetadataRef.current.expanded) : []),
  );
  const [activeStrategyId, setActiveStrategyId] = useState<string>(
    () => initialMetadataRef.current?.activeId ?? '',
  );
  const persistedMetadataSignatureRef = useRef<string>(initialMetadataRef.current?.signature ?? '');
  const lastAppliedStrategyRef = useRef<string | null>(null);
  const lastModelTypeRef = useRef<string | null>(null);
  const activeStrategy = useMemo(() => {
    if (!strategies.length) {
      return null;
    }
    const explicit = strategies.find((strategy) => strategy.id === activeStrategyId);
    return explicit ?? strategies[0];
  }, [activeStrategyId, strategies]);

  const persistedMetadataFromConfig = useMemo(
    () => convertConfigToPersistedMetadata(draftConfigState),
    [draftConfigState],
  );
  const configMetadataSignature = useMemo(
    () => buildMetadataSignature(persistedMetadataFromConfig),
    [persistedMetadataFromConfig],
  );

  useEffect(() => {
    if (configMetadataSignature === persistedMetadataSignatureRef.current) {
      return;
    }

    const nextStrategies = persistedMetadataFromConfig.strategies.length
      ? persistedMetadataFromConfig.strategies.map(buildStrategyDraftFromPersisted)
      : [];
    const nextActiveId = resolveActiveStrategyId(nextStrategies, persistedMetadataFromConfig.activeId);
    const nextExpandedSet = resolveExpandedStrategyIds(nextStrategies, persistedMetadataFromConfig.expandedIds);
    if (nextActiveId && !nextExpandedSet.has(nextActiveId) && nextStrategies.length) {
      nextExpandedSet.add(nextActiveId);
    }

    lastAppliedStrategyRef.current = null;
    setStrategies(nextStrategies);
    setActiveStrategyId(nextActiveId);
    setExpandedStrategyIds(nextExpandedSet);
    persistedMetadataSignatureRef.current = configMetadataSignature;
  }, [
    configMetadataSignature,
    persistedMetadataFromConfig,
    draftConfigState,
  ]);

  useEffect(() => {
    if (!strategies.length) {
      return;
    }
    const hasActive = strategies.some((strategy) => strategy.id === activeStrategyId);
    if (!hasActive) {
      setActiveStrategyId(strategies[0].id);
    }
  }, [activeStrategyId, strategies]);

  useEffect(() => {
    const metadata = convertDraftStateToPersistedMetadata(strategies, activeStrategyId, expandedStrategyIds);
    const nextSignature = buildMetadataSignature(metadata);
    if (persistedMetadataSignatureRef.current === nextSignature) {
      return;
    }

    setDraftConfigState((previous) => {
      const previousMetadata = convertConfigToPersistedMetadata(previous);
      const previousSignature = buildMetadataSignature(previousMetadata);
      if (previousSignature === nextSignature) {
        persistedMetadataSignatureRef.current = previousSignature;
        return previous ?? {};
      }

      const nextState = previous ? { ...previous } : {};

      if (metadata.strategies.length) {
        nextState[TUNING_STRATEGIES_KEY] = metadata.strategies.map((entry) => ({
          ...entry,
          config: cloneJson(entry.config),
          searchSpaceTexts: { ...entry.searchSpaceTexts },
          searchSpaceFieldErrors: { ...entry.searchSpaceFieldErrors },
        }));
      } else {
        delete nextState[TUNING_STRATEGIES_KEY];
      }

      if (metadata.activeId) {
        nextState[TUNING_ACTIVE_ID_KEY] = metadata.activeId;
      } else {
        delete nextState[TUNING_ACTIVE_ID_KEY];
      }

      if (metadata.expandedIds.length) {
        nextState[TUNING_EXPANDED_KEY] = [...metadata.expandedIds];
      } else {
        delete nextState[TUNING_EXPANDED_KEY];
      }

      persistedMetadataSignatureRef.current = nextSignature;
      return nextState;
    });
  }, [activeStrategyId, expandedStrategyIds, setDraftConfigState, strategies]);

  useEffect(() => {
    if (!activeStrategy) {
      return;
    }
    if (lastAppliedStrategyRef.current === activeStrategy.id) {
      return;
    }
    lastAppliedStrategyRef.current = activeStrategy.id;
    const nextConfig = cloneConfigForStrategy(activeStrategy.config);
    setDraftConfigState((previous) => {
      const preservedMetadata = extractStoredStrategyMetadataFields(previous);
      return {
        ...preservedMetadata,
        ...nextConfig,
      };
    });
    setSearchSpaceTexts(activeStrategy.searchSpaceTexts ?? {});
    setSearchSpaceFieldErrors(activeStrategy.searchSpaceFieldErrors ?? {});
    setHasAutoFilledSearchSpace(activeStrategy.hasAutoFilledSearchSpace ?? false);
    setSearchSpaceManuallyCleared(activeStrategy.searchSpaceManuallyCleared ?? false);
    const strategyModelType = typeof nextConfig?.[modelTypeParameterName] === 'string'
      ? String(nextConfig[modelTypeParameterName])
      : (runtimeConfig?.modelType ?? '');
    lastModelTypeRef.current = strategyModelType;
  }, [activeStrategy, modelTypeParameterName, runtimeConfig?.modelType, setDraftConfigState]);

  useEffect(() => {
    if (!activeStrategy) {
      return;
    }
    const nextConfig = cloneConfigForStrategy(draftConfigState);
    const configChanged = JSON.stringify(nextConfig) !== JSON.stringify(activeStrategy.config ?? {});
    const textsChanged = !shallowEqualRecord(searchSpaceTexts, activeStrategy.searchSpaceTexts);
    const errorsChanged = !shallowEqualRecord(searchSpaceFieldErrors, activeStrategy.searchSpaceFieldErrors);
    const autoFillChanged = activeStrategy.hasAutoFilledSearchSpace !== hasAutoFilledSearchSpace;
    const clearedChanged = activeStrategy.searchSpaceManuallyCleared !== searchSpaceManuallyCleared;

    if (!configChanged && !textsChanged && !errorsChanged && !autoFillChanged && !clearedChanged) {
      return;
    }

    setStrategies((previous) =>
      previous.map((strategy) => {
        if (strategy.id !== activeStrategy.id) {
          return strategy;
        }
        return {
          ...strategy,
          config: nextConfig,
          searchSpaceTexts: { ...searchSpaceTexts },
          searchSpaceFieldErrors: { ...searchSpaceFieldErrors },
          hasAutoFilledSearchSpace,
          searchSpaceManuallyCleared,
        };
      }),
    );
  }, [
    activeStrategy,
    draftConfigState,
    hasAutoFilledSearchSpace,
    searchSpaceFieldErrors,
    searchSpaceManuallyCleared,
    searchSpaceTexts,
  ]);

  const handleAddStrategy = useCallback(() => {
    const nextOrdinal = strategies.length + 1;
    const baseConfig = cloneConfigForStrategy(draftConfigState);
    const usedStrategies = new Set<SearchStrategy>();
    strategies.forEach((strategy) => {
      const normalized = normalizeSearchStrategyValue(
        strategy.config?.[searchStrategyParameterName],
        availableSearchStrategies,
      );
      if (normalized) {
        usedStrategies.add(normalized);
      }
    });

    const nextStrategyValue = availableSearchStrategies.find((candidate) => !usedStrategies.has(candidate)) ?? searchStrategy;

    const configCloneRaw = cloneJson(baseConfig);
    const configWithStrategy: Record<string, any> = isRecord(configCloneRaw) ? configCloneRaw : {};
    configWithStrategy[searchStrategyParameterName] = nextStrategyValue;

    if (nextStrategyValue === 'random' || nextStrategyValue === 'optuna') {
      const iterationsValue = searchIterations !== null ? searchIterations : searchIterationsDefault;
      if (iterationsValue !== null) {
        configWithStrategy[searchIterationsParameterName] = iterationsValue;
      } else {
        delete configWithStrategy[searchIterationsParameterName];
      }

      const randomStateValue = searchRandomState !== null ? searchRandomState : searchRandomStateDefault;
      if (randomStateValue !== null) {
        configWithStrategy[searchRandomStateParameterName] = randomStateValue;
      } else {
        delete configWithStrategy[searchRandomStateParameterName];
      }
    } else {
      delete configWithStrategy[searchIterationsParameterName];
      delete configWithStrategy[searchRandomStateParameterName];
    }

    const newStrategy = createStrategyDraft(createStrategyLabel(nextOrdinal), configWithStrategy, {
      searchSpaceTexts: { ...searchSpaceTexts },
      searchSpaceFieldErrors: {},
      hasAutoFilledSearchSpace,
      searchSpaceManuallyCleared,
    });

    setStrategies((previous) => [...previous, newStrategy]);
    setExpandedStrategyIds((prev) => new Set([...prev, newStrategy.id]));
    lastAppliedStrategyRef.current = null;
    setActiveStrategyId(newStrategy.id);
  }, [
    availableSearchStrategies,
    draftConfigState,
    hasAutoFilledSearchSpace,
    searchIterations,
    searchIterationsDefault,
    searchRandomState,
    searchRandomStateDefault,
    searchSpaceManuallyCleared,
    searchSpaceTexts,
    searchStrategy,
    searchStrategyParameterName,
    searchIterationsParameterName,
    searchRandomStateParameterName,
    setStrategies,
    setActiveStrategyId,
    setExpandedStrategyIds,
    strategies,
  ]);

  const handleToggleStrategy = useCallback((strategyId: string) => {
    setExpandedStrategyIds((prev) => {
      const next = new Set(prev);
      if (next.has(strategyId)) {
        next.delete(strategyId);
      } else {
        next.add(strategyId);
      }
      return next;
    });
  }, []);

  const handleActivateStrategy = useCallback(
    (strategyId: string) => {
      if (!strategyId || strategyId === activeStrategyId) {
        return;
      }
      lastAppliedStrategyRef.current = null;
      setActiveStrategyId(strategyId);
      setExpandedStrategyIds((previous) => {
        const next = new Set(previous);
        next.add(strategyId);
        return next;
      });
    },
    [activeStrategyId],
  );

  const handleRemoveStrategy = useCallback(
    (strategyId: string) => {
      let nextActiveId: string | null = null;
      setStrategies((previous) => {
        if (previous.length <= 1) {
          return previous;
        }
        const filtered = previous.filter((strategy) => strategy.id !== strategyId);
        if (!filtered.length) {
          return previous;
        }
        const relabeled = filtered.map((strategy, index) => ({
          ...strategy,
          label: createStrategyLabel(index + 1),
        }));
        if (strategyId === activeStrategyId) {
          lastAppliedStrategyRef.current = null;
          const fallbackId = relabeled[0].id;
          nextActiveId = fallbackId;
          setActiveStrategyId(fallbackId);
        }
        return relabeled;
      });
      setExpandedStrategyIds((prev) => {
        const next = new Set(prev);
        next.delete(strategyId);
        if (nextActiveId) {
          next.add(nextActiveId);
        }
        return next;
      });
    },
    [activeStrategyId],
  );

  const strategySummaries = useMemo(() => {
    return strategies.map((strategy, index) => {
      const config = strategy.config ?? {};
      const modelValueRaw = config?.[modelTypeParameterName];
      const modelValue = typeof modelValueRaw === 'string' ? modelValueRaw.trim() : '';
      const modelLabel = modelTypeOptions.find((option) => option.value === modelValue)?.label ?? modelValue;

      const searchStrategyValueRaw = config?.[searchStrategyParameterName];
      const searchStrategyValue = typeof searchStrategyValueRaw === 'string' ? searchStrategyValueRaw.trim() : '';
      const searchLabel = (searchStrategyParameter?.options ?? []).find((option) => option.value === searchStrategyValue)?.label;

      const scoringValueRaw = config?.[scoringMetricParameterName];
      const scoringValue = typeof scoringValueRaw === 'string' ? scoringValueRaw.trim() : '';

      const baselineParsed = parseJsonObject(config?.baseline_hyperparameters);
      const baselineError = baselineParsed.error;

      const searchParse = parseJsonObject(config?.search_space);
      const normalized = normalizeSearchSpace(searchParse.value);
      const parameterCount = normalized.value ? Object.keys(normalized.value).length : 0;
      const candidateCount = normalized.value
        ? Object.values(normalized.value).reduce((total, values) =>
            total + (Array.isArray(values) ? values.length : 0),
          0)
        : 0;
      const hasErrors = baselineError !== null || Object.values(strategy.searchSpaceFieldErrors).some((message) => Boolean(message));

      return {
        id: strategy.id,
        label: strategy.label || createStrategyLabel(index + 1),
        modelLabel: modelLabel || 'Select model',
        searchStrategyLabel: searchLabel ?? (searchStrategyValue || 'Random search'),
        scoringLabel: scoringValue || 'Estimator default',
        parameterCount,
        candidateCount,
        hasErrors,
      };
    });
  }, [
    modelTypeOptions,
    modelTypeParameterName,
    scoringMetricParameterName,
    searchStrategyParameter,
    searchStrategyParameterName,
    strategies,
  ]);

  useEffect(() => {
    const previous = lastModelTypeRef.current;
    if (previous === null) {
      lastModelTypeRef.current = modelType;
      return;
    }
    if (previous === modelType) {
      return;
    }
    lastModelTypeRef.current = modelType;
    setSearchSpaceTexts({});
    setSearchSpaceFieldErrors({});
    setHasAutoFilledSearchSpace(false);
    setSearchSpaceManuallyCleared(false);
  }, [modelType]);

  useEffect(() => {
    if (!modelType || !hyperparamFields.length) {
      return;
    }
    setSearchSpaceTexts((previous) => {
      const next = { ...previous };
      let changed = false;
      hyperparamFields.forEach((field) => {
        if (searchSpaceFieldErrors[field.name]) {
          return;
        }
        const values = filteredSearchOverrides[field.name];
        const formatted =
          Array.isArray(values) && values.length > 0
            ? values.map((value) => formatSearchValue(field, value)).join(', ')
            : '';
        if (next[field.name] !== formatted) {
          next[field.name] = formatted;
          changed = true;
        }
      });
      return changed ? next : previous;
    });
  }, [filteredSearchOverrides, hyperparamFields, modelType, searchSpaceFieldErrors]);

  useEffect(() => {
    if (!modelType || !hyperparamFields.length) {
      return;
    }

    if (searchSpaceManuallyCleared) {
      return;
    }

    if (hasSearchSpaceEntries) {
      if (!hasAutoFilledSearchSpace) {
        setHasAutoFilledSearchSpace(true);
      }
      return;
    }

    if (hasAutoFilledSearchSpace) {
      return;
    }

    const generated = buildDefaultSearchSpace(hyperparamFields, hyperparamDefaults);
    if (!Object.keys(generated).length) {
      return;
    }

    setHasAutoFilledSearchSpace(true);
    setSearchSpaceFieldErrors({});
    setSearchSpaceTexts(() => {
      const next: Record<string, string> = {};
      hyperparamFields.forEach((field) => {
        if (!generated[field.name]) {
          return;
        }
        const formatted = generated[field.name]
          .map((value) => formatSearchValue(field, value))
          .filter((entry) => entry.length > 0)
          .join(', ');
        next[field.name] = formatted;
      });
      return next;
    });
    setDraftConfigState((previous) => {
      const currentState = previous ? { ...previous } : {};
      currentState.search_space = generated;
      return currentState;
    });
  }, [
    hasAutoFilledSearchSpace,
    hyperparamDefaults,
    hyperparamFields,
    modelType,
    hasSearchSpaceEntries,
    searchSpaceManuallyCleared,
    setDraftConfigState,
  ]);

  const [scoringDraft, setScoringDraft] = useState(scoringMetric ?? '');

  useEffect(() => {
    setScoringDraft(scoringMetric ?? '');
  }, [scoringMetric]);

  const scoringSuggestions = useMemo(
    () => SCORING_SUGGESTIONS[problemType === 'regression' ? 'regression' : 'classification'],
    [problemType],
  );

  const scoringLabel = scoringMetricParameter?.label ?? 'Scoring metric';
  const scoringDescription = scoringMetricParameter?.description ?? 'Pick the evaluation metric used to rank candidates.';
  const scoringInputId = useMemo(() => `tuning-${nodeId}-scoring-metric`, [nodeId]);

  const hasSearchSpaceFieldErrors = useMemo(
    () => Object.values(searchSpaceFieldErrors).some((message) => Boolean(message)),
    [searchSpaceFieldErrors],
  );

  const updateBaselineValue = useCallback(
    (fieldName: string, nextValue: any) => {
      if (!allowedHyperparamNames.has(fieldName)) {
        return;
      }

      setDraftConfigState((previous) => {
        const nextState = { ...previous };
        const parsed = parseJsonObject(previous?.baseline_hyperparameters).value ?? {};
        const filtered = filterHyperparametersByFields(parsed, allowedHyperparamNames);
        const updated = { ...filtered };

        if (nextValue === undefined) {
          delete updated[fieldName];
        } else {
          updated[fieldName] = nextValue;
        }

        if (Object.keys(updated).length > 0) {
          nextState.baseline_hyperparameters = updated;
        } else {
          delete nextState.baseline_hyperparameters;
        }

        return nextState;
      });
    },
    [allowedHyperparamNames, setDraftConfigState],
  );

  const handleSearchSpaceInputChange = useCallback(
    (field: ModelHyperparameterField, rawText: string) => {
      setSearchSpaceTexts((previous) => ({
        ...previous,
        [field.name]: rawText,
      }));

      const { values, error } = parseSearchInput(field, rawText);
      setSearchSpaceFieldErrors((previous) => ({
        ...previous,
        [field.name]: error,
      }));

      if (error) {
        return;
      }

      setDraftConfigState((previous) => {
        const nextState = { ...previous };
        const parsed = parseJsonObject(previous?.search_space).value ?? {};
        const filtered = filterSearchSpaceByFields(parsed, allowedHyperparamNames);
        const updated = { ...filtered };

        if (values.length > 0) {
          updated[field.name] = values.map((value) => cloneJson(value));
        } else {
          delete updated[field.name];
        }

        if (Object.keys(updated).length > 0) {
          nextState.search_space = updated;
        } else {
          delete nextState.search_space;
        }

        return nextState;
      });
    },
    [allowedHyperparamNames, setDraftConfigState],
  );

  const clearBaselineValue = useCallback(
    (fieldName: string) => {
      updateBaselineValue(fieldName, undefined);
    },
    [updateBaselineValue],
  );

  const handleResetBaselineOverrides = useCallback(() => {
    setDraftConfigState((previous) => {
      if (!previous || !previous.baseline_hyperparameters) {
        return previous ?? {};
      }
      const nextState = { ...previous };
      delete nextState.baseline_hyperparameters;
      return nextState;
    });
  }, [setDraftConfigState]);

  const handleClearSearchSpace = useCallback(() => {
    setSearchSpaceManuallyCleared(true);
    setHasAutoFilledSearchSpace(true);
    setSearchSpaceTexts({});
    setSearchSpaceFieldErrors({});
    setDraftConfigState((previous) => {
      if (!previous || !previous.search_space) {
        return previous ?? {};
      }
      const nextState = { ...previous };
      delete nextState.search_space;
      return nextState;
    });
  }, [setDraftConfigState]);

  const handleScoringChange = useCallback(
    (value: string) => {
      setScoringDraft(value);
      const trimmed = value.trim();
      setDraftConfigState((previous) => {
        const nextState = { ...previous };
        if (trimmed.length > 0) {
          nextState.scoring_metric = trimmed;
        } else {
          delete nextState.scoring_metric;
        }
        return nextState;
      });
    },
    [setDraftConfigState],
  );

  const renderBaselineField = useCallback(
    (field: ModelHyperparameterField) => {
      if (!field?.name || !allowedHyperparamNames.has(field.name)) {
        return null;
      }

      const fieldId = `baseline-${nodeId}-${field.name}`;
      const override = filteredBaselineOverrides[field.name];
      const hasOverride = override !== undefined;
      const defaultValue = hyperparamDefaults?.[field.name];
      const label = field.label ?? field.name;
      const description = field.description;
      const defaultHint = defaultValue !== undefined
        ? `Default: ${formatValuePreview(defaultValue)}`
        : 'Default: estimator setting';

      let control: React.ReactNode;

      if (field.type === 'number') {
        const numericValue = typeof override === 'number' ? String(override) : '';
        control = (
          <input
            id={fieldId}
            type="number"
            className="canvas-modal__parameter-input"
            value={numericValue}
            onChange={(event) => {
              const raw = event.target.value;
              if (!raw.length) {
                clearBaselineValue(field.name);
                return;
              }
              const parsed = Number(raw);
              if (Number.isFinite(parsed)) {
                updateBaselineValue(field.name, parsed);
              }
            }}
            min={field.min}
            max={field.max}
            step={field.step}
            placeholder="Leave blank to use default"
          />
        );
      } else if (field.type === 'select') {
        const selectValue = hasOverride ? String(override) : '';
        control = (
          <select
            id={fieldId}
            className="canvas-modal__parameter-select"
            value={selectValue}
            onChange={(event) => {
              const next = event.target.value;
              if (!next.length) {
                clearBaselineValue(field.name);
                return;
              }
              updateBaselineValue(field.name, next);
            }}
          >
            <option value="">Use default</option>
            {(field.options ?? []).map((option) => (
              <option key={String(option.value)} value={String(option.value)}>
                {option.label}
              </option>
            ))}
          </select>
        );
      } else if (field.type === 'boolean') {
        const selectValue = hasOverride ? (override ? 'true' : 'false') : '';
        control = (
          <select
            id={fieldId}
            className="canvas-modal__parameter-select"
            value={selectValue}
            onChange={(event) => {
              const next = event.target.value;
              if (!next.length) {
                clearBaselineValue(field.name);
                return;
              }
              updateBaselineValue(field.name, next === 'true');
            }}
          >
            <option value="">Use default</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        );
      } else {
        const textValue = typeof override === 'string' ? override : '';
        control = (
          <input
            id={fieldId}
            type="text"
            className="canvas-modal__parameter-input"
            value={textValue}
            onChange={(event) => {
              const next = event.target.value;
              if (!next.length) {
                clearBaselineValue(field.name);
                return;
              }
              updateBaselineValue(field.name, next);
            }}
            placeholder="Leave blank to use default"
          />
        );
      }

      return (
        <div key={field.name} className="canvas-modal__parameter">
          <label htmlFor={fieldId} className="canvas-modal__parameter-label">
            {label}
            {description && (
              <span className="canvas-modal__parameter-description">{description}</span>
            )}
          </label>
          {control}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginTop: '0.4rem',
              gap: '0.75rem',
            }}
          >
            <small style={{ color: 'rgba(148, 163, 184, 0.85)' }}>
              {hasOverride
                ? `Override: ${formatValuePreview(override)}`
                : defaultHint}
            </small>
          </div>
        </div>
      );
    },
    [allowedHyperparamNames, filteredBaselineOverrides, hyperparamDefaults, nodeId, updateBaselineValue],
  );

  const renderSearchSpaceField = useCallback(
    (field: ModelHyperparameterField) => {
      if (!field?.name || !allowedHyperparamNames.has(field.name)) {
        return null;
      }

      const fieldId = `search-${nodeId}-${field.name}`;
      const textValue = searchSpaceTexts[field.name] ?? '';
      const configuredValues = filteredSearchOverrides[field.name];
      const configuredCount = Array.isArray(configuredValues) ? configuredValues.length : 0;
      const errorMessage = searchSpaceFieldErrors[field.name];
      const defaultValue = hyperparamDefaults?.[field.name];

      const placeholder = (() => {
        if (field.type === 'number') {
          return 'Comma or newline separated numbers (e.g. 0.01, 0.1, 1)';
        }
        if (field.type === 'boolean') {
          return 'Enter true/false values, separated by commas or new lines';
        }
        if (field.type === 'select') {
          return 'Comma or newline separated option values';
        }
        return 'Comma or newline separated values';
      })();

      const previewValues = Array.isArray(configuredValues)
        ? configuredValues.slice(0, 3).map((value) => formatValuePreview(value)).join(', ')
        : '';

      return (
        <div key={field.name} className="canvas-modal__parameter">
          <label htmlFor={fieldId} className="canvas-modal__parameter-label">
            {field.label ?? field.name}
            {field.description && (
              <span className="canvas-modal__parameter-description">{field.description}</span>
            )}
          </label>
          <textarea
            id={fieldId}
            className="canvas-modal__parameter-input"
            rows={3}
            value={textValue}
            onChange={(event) => handleSearchSpaceInputChange(field, event.target.value)}
            placeholder={placeholder}
            style={{ minHeight: '92px', resize: 'vertical' }}
          />
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginTop: '0.4rem',
              gap: '0.75rem',
              flexWrap: 'wrap',
            }}
          >
            <small style={{ color: 'rgba(148, 163, 184, 0.85)' }}>
              {configuredCount > 0
                ? `Configured ${configuredCount} value${configuredCount === 1 ? '' : 's'}${previewValues ? ` (${previewValues}${configuredCount > 3 ? '…' : ''})` : ''}`
                : `Default: ${formatValuePreview(defaultValue)}`}
            </small>
          </div>
          {errorMessage && (
            <p className="canvas-modal__note canvas-modal__note--error" style={{ marginTop: '0.5rem' }}>
              {errorMessage}
            </p>
          )}
        </div>
      );
    },
    [allowedHyperparamNames, filteredSearchOverrides, handleSearchSpaceInputChange, hyperparamDefaults, nodeId, searchSpaceFieldErrors, searchSpaceTexts],
  );

  const searchSpaceDimension = useMemo(() => {
    if (!hasSearchSpaceEntries) {
      return { parameterCount: 0, candidateProduct: null as number | null };
    }
    const entries = Object.values(filteredSearchOverrides);
    const parameterCount = entries.length;
    let candidateProduct: number | null = null;
    if (entries.length > 0) {
      let product = 1;
      let overflow = false;
      entries.forEach((values) => {
        product *= Math.max(values.length, 1);
        if (!Number.isFinite(product) || product > 1_000_000) {
          overflow = true;
        }
      });
      candidateProduct = overflow ? null : product;
    }
    return { parameterCount, candidateProduct };
  }, [filteredSearchOverrides, hasSearchSpaceEntries]);

  const graphPayload = useMemo<FeatureGraph | null>(() => {
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
      return null;
    }
    return {
      nodes: graph.nodes,
      edges: graph.edges,
    };
  }, [graph]);

  const mergeConfigIntoGraph = useCallback(
    (configToMerge: Record<string, any> | null | undefined): FeatureGraph | null => {
      if (!graphPayload) {
        return null;
      }
      const clonedGraph = cloneJson(graphPayload);
      if (!clonedGraph || !Array.isArray(clonedGraph.nodes)) {
        return graphPayload;
      }

      const sanitizedConfig = stripTuningMetadata(configToMerge);
      const hasSanitizedConfig = Object.keys(sanitizedConfig).length > 0;

      clonedGraph.nodes = clonedGraph.nodes.map((node: any) => {
        if (!node || node.id !== nodeId) {
          return node;
        }
        const existingData = node?.data ?? {};
        const existingConfig = existingData?.config ?? {};
        const sanitizedExistingConfig = stripTuningMetadata(existingConfig);
        const mergedConfig = hasSanitizedConfig
          ? { ...sanitizedExistingConfig, ...sanitizedConfig }
          : sanitizedExistingConfig;

        return {
          ...node,
          data: {
            ...existingData,
            config: mergedConfig,
            isConfigured: true,
          },
        };
      });

      return clonedGraph;
    },
    [graphPayload, nodeId],
  );

  const graphWithDraftConfig = useMemo<FeatureGraph | null>(() => {
    const merged = mergeConfigIntoGraph(draftConfigState);
    if (merged) {
      return merged;
    }
    return graphPayload;
  }, [draftConfigState, graphPayload, mergeConfigIntoGraph]);

  const validationConnectionStatus = useMemo(() => {
    if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
      return null;
    }

    const nodes = graph.nodes as any[];
    const edges = graph.edges as any[];

    const connectedSplits = (() => {
      const currentNode = nodes.find((entry: any) => entry?.id === nodeId);
      if (!currentNode) {
        return [] as string[];
      }
      return sanitizeSplitList(currentNode?.data?.connectedSplits);
    })();

    if (connectedSplits.includes('validation')) {
      return true;
    }

    const hasValidationEdge = edges.some((edge: any) => {
      if (!edge) {
        return false;
      }
      const targetId = typeof edge.target === 'string' ? edge.target : null;
      if (targetId !== nodeId) {
        return false;
      }
      const sourceHandle = typeof edge.sourceHandle === 'string' ? edge.sourceHandle.toLowerCase() : '';
      return sourceHandle.includes('validation');
    });

    if (hasValidationEdge) {
      return true;
    }

    const upstreamNodesExposeValidation = edges.some((edge: any) => {
      if (!edge) {
        return false;
      }
      const targetId = typeof edge.target === 'string' ? edge.target : null;
      if (targetId !== nodeId) {
        return false;
      }
      const sourceId = typeof edge.source === 'string' ? edge.source : null;
      if (!sourceId) {
        return false;
      }
      const upstreamNode = nodes.find((entry: any) => entry?.id === sourceId);
      if (!upstreamNode) {
        return false;
      }
      const activeSplits = sanitizeSplitList(upstreamNode?.data?.activeSplits);
      return activeSplits.includes('validation');
    });

    if (!upstreamNodesExposeValidation) {
      return null;
    }

    return false;
  }, [graph, nodeId]);

  useEffect(() => {
    let cancelled = false;
    if (!sourceId || !graphPayload) {
      setPipelineIdFromSavedConfig(null);
      return () => {
        cancelled = true;
      };
    }

    generatePipelineId(sourceId, graphPayload)
      .then((value) => {
        if (!cancelled) {
          setPipelineIdFromSavedConfig(value);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPipelineIdFromSavedConfig(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [graphPayload, sourceId]);

  useEffect(() => {
    let cancelled = false;
    setPipelineError(null);

    const graphForHash = graphWithDraftConfig ?? graphPayload;
    if (!sourceId || !graphForHash) {
      setPipelineId(null);
      return () => {
        cancelled = true;
      };
    }

    generatePipelineId(sourceId, graphForHash)
      .then((value) => {
        if (!cancelled) {
          setPipelineId(value);
        }
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setPipelineError(error?.message ?? 'Unable to compute pipeline ID');
          setPipelineId(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [graphPayload, graphWithDraftConfig, sourceId]);

  useEffect(() => {
    if (!pipelineId || !pipelineIdFromSavedConfig) {
      setHasDraftChanges(false);
      return;
    }
    setHasDraftChanges(pipelineId !== pipelineIdFromSavedConfig);
  }, [pipelineId, pipelineIdFromSavedConfig]);

  const shouldFetchJobs = Boolean(sourceId && nodeId);
  const tuningJobsQueryKey = useMemo(() => {
    return ['feature-canvas', 'hyperparameter-tuning-jobs', sourceId ?? 'none', pipelineId ?? 'pending', nodeId];
  }, [nodeId, pipelineId, sourceId]);

  const tuningJobsQuery = useQuery<HyperparameterTuningJobListResponse, Error>({
    queryKey: tuningJobsQueryKey,
    queryFn: async () => {
      const baseParams: FetchHyperparameterTuningJobsOptions = {
        datasetSourceId: sourceId || undefined,
        nodeId,
        limit: 5,
      };

      if (pipelineId) {
        const scoped = await fetchHyperparameterTuningJobs({ ...baseParams, pipelineId });
        if ((scoped?.jobs?.length ?? 0) > 0) {
          return scoped;
        }
      }

      return fetchHyperparameterTuningJobs(baseParams);
    },
    enabled: shouldFetchJobs,
    retry: (failureCount, error) => {
      if (error.message.includes('Sign in')) {
        return false;
      }
      return failureCount < 2;
    },
    refetchInterval: (query) => {
      const currentData = query.state.data as HyperparameterTuningJobListResponse | undefined;
      const jobList = currentData?.jobs ?? [];
      const hasActiveJob = jobList.some((job: HyperparameterTuningJobSummary) => job.status === 'queued' || job.status === 'running');
      return hasActiveJob ? 1000 : false;
    },
  });

  const {
    mutateAsync: enqueueTuningJob,
    isPending: isCreatingJob,
    error: createJobError,
  } = useMutation({
    mutationFn: createHyperparameterTuningJob,
    onSuccess: (result) => {
      const jobCount = Array.isArray(result.jobs) ? result.jobs.length : 0;
      const firstJob = jobCount > 0 ? result.jobs[0] : null;
      setLastCreatedJob(firstJob);
      setLastCreatedJobCount(jobCount);
      queryClient.invalidateQueries({ queryKey: tuningJobsQueryKey });
      if (firstJob && onUpdateNodeData) {
        onUpdateNodeData(nodeId, { backgroundExecutionStatus: 'loading' });
      }
    },
  });

  useEffect(() => {
    if (!tuningJobsQuery.data || !onUpdateNodeData) {
      return;
    }
    const jobs = tuningJobsQuery.data.jobs ?? [];
    if (jobs.length === 0) {
      return;
    }
    // The API returns jobs sorted by created_at desc, so the first one is the latest.
    const latestJob = jobs[0];
    const status = latestJob.status?.toLowerCase();
    let nodeStatus = 'idle';

    if (status === 'running' || status === 'queued') {
      nodeStatus = 'loading';
    } else if (status === 'succeeded') {
      nodeStatus = 'success';
    } else if (status === 'failed' || status === 'cancelled') {
      nodeStatus = 'error';
    }

    const progress = typeof latestJob.progress === 'number' ? latestJob.progress : undefined;

    if (nodeStatus !== currentStatus || (progress !== undefined && progress !== currentProgress)) {
      onUpdateNodeData(nodeId, { backgroundExecutionStatus: nodeStatus, progress });
    }
  }, [tuningJobsQuery.data, onUpdateNodeData, nodeId, currentStatus, currentProgress]);

  const prerequisites = useMemo(() => {
    const notes: string[] = [];
    if (!sourceId) {
      notes.push('Select a dataset before launching tuning jobs.');
    }
    if (!graphPayload || !graphPayload.nodes.length) {
      notes.push('Connect this node to an upstream pipeline before tuning.');
    }
    if (!targetColumn) {
      notes.push('Set a target column before tuning.');
    }
    if (validationConnectionStatus === false) {
      notes.push('Connect the validation split to this node to enable tuning.');
    }
    if (!modelType) {
      notes.push('Choose a model type before enqueuing tuning jobs.');
    }
    if (strategies.length === 0) {
      notes.push('Add a tuning strategy before launching tuning jobs.');
    }
    if (!hasSearchSpaceEntries) {
      notes.push('Define at least one hyperparameter in the search space.');
    }
    if (baselineError) {
      notes.push(baselineError);
    }
    if (searchSpaceError) {
      notes.push(searchSpaceError);
    }
    if (hasSearchSpaceFieldErrors) {
      notes.push('Resolve invalid candidate values in the search space.');
    }
    if ((searchStrategy === 'random' || searchStrategy === 'optuna') && !searchIterations) {
      notes.push('Provide the maximum iterations for random or Optuna search.');
    }
    if (cvEnabled && cvFolds < 2) {
      notes.push('Cross-validation requires at least 2 folds.');
    }
    if (pipelineError) {
      notes.push(pipelineError);
    }
    return notes;
  }, [
    baselineError,
    cvEnabled,
    cvFolds,
    graphPayload,
    hasSearchSpaceEntries,
    hasSearchSpaceFieldErrors,
    modelType,
    pipelineError,
    searchIterations,
    searchSpaceError,
    searchStrategy,
    sourceId,
    targetColumn,
    validationConnectionStatus,
    strategies.length,
  ]);

  const tuningJobs = tuningJobsQuery.data?.jobs ?? [];
  const isJobsLoading = tuningJobsQuery.isLoading || tuningJobsQuery.isFetching;
  const jobsError = tuningJobsQuery.error as Error | null;

  const hasScalingConvergenceSignals = useMemo(() => {
    if (createJobError instanceof Error && hasScalingConvergenceMessage(createJobError.message)) {
      return true;
    }
    if (batchEnqueueError && hasScalingConvergenceMessage(batchEnqueueError)) {
      return true;
    }
    if (lastCreatedJob && detectScalingConvergenceFromJob(lastCreatedJob)) {
      return true;
    }
    if (tuningJobs.some((job) => detectScalingConvergenceFromJob(job))) {
      return true;
    }
    return false;
  }, [batchEnqueueError, createJobError, lastCreatedJob, tuningJobs]);

  const scalingWarning = useScalingWarning({
    graph,
    nodeId,
    modelType,
    problemType,
    modelTypeOptions,
    enabled: hasScalingConvergenceSignals,
  });

  useEffect(() => {
    if (!scalingWarning) {
      setShowScalingDetails(false);
    }
  }, [scalingWarning]);

  const isActionDisabled =
    prerequisites.length > 0 ||
    hasSearchSpaceFieldErrors ||
    !pipelineId ||
    !sourceId ||
    !graphPayload ||
    strategies.length === 0 ||
    !hasSearchSpaceEntries ||
    isCreatingJob;

  const filteredModelTypeParameter = useMemo(() => {
    if (!modelTypeParameter) {
      return null;
    }
    if (!modelTypeOptions.length) {
      return modelTypeParameter;
    }
    return {
      ...modelTypeParameter,
      options: modelTypeOptions,
    };
  }, [modelTypeOptions, modelTypeParameter]);

  const handleEnqueueTuning = useCallback(async () => {
    if (!pipelineId || !sourceId || !graphPayload || strategies.length === 0) {
      return;
    }

    if (onSaveDraftConfig) {
      try {
        const maybeResult = onSaveDraftConfig({ closeModal: false });
        setHasDraftChanges(false);
        if (maybeResult && typeof (maybeResult as Promise<unknown>).then === 'function') {
          await (maybeResult as Promise<unknown>);
        }
      } catch (error) {
        // ignore save errors here; enqueue API will surface problems if relevant
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const buildStrategyPlans = async (): Promise<{
      plans: Array<{ payload: HyperparameterTuningJobCreatePayload; label: string }>;
      issues: string[];
    }> => {
      const plans: Array<{ payload: HyperparameterTuningJobCreatePayload; label: string }> = [];
      const issues: string[] = [];

      const resolveAllowedNamesForModel = async (
        resolvedModelType: string,
      ): Promise<Set<string>> => {
        if (!resolvedModelType) {
          return new Set<string>();
        }

        if (resolvedModelType === modelType) {
          return allowedHyperparamNames;
        }

        const queryKey = ['model-hyperparameters', resolvedModelType] as const;
        const cached = queryClient.getQueryData<ModelHyperparametersResponse>(queryKey);
        const collectNames = (fields: ModelHyperparameterField[] | undefined | null) => {
          const names = new Set<string>();
          (fields ?? []).forEach((field) => {
            if (field?.name) {
              names.add(field.name);
            }
          });
          return names;
        };

        if (cached) {
          return collectNames(cached.fields);
        }

        try {
          const result = await queryClient.fetchQuery<ModelHyperparametersResponse>({
            queryKey,
            queryFn: () => fetchModelHyperparameters(resolvedModelType),
          });
          return collectNames(result?.fields);
        } catch (error) {
          return new Set<string>();
        }
      };

      for (let index = 0; index < strategies.length; index += 1) {
        const strategy = strategies[index];
        const strategyLabel = strategy.label || createStrategyLabel(index + 1);
        const strategyConfig = cloneConfigForStrategy(strategy.config);

        const rawModelType = strategyConfig?.[modelTypeParameterName];
        const resolvedModelType =
          typeof rawModelType === 'string' && rawModelType.trim().length
            ? rawModelType.trim()
            : modelType;

        if (!resolvedModelType) {
          issues.push(`${strategyLabel} is missing a model selection.`);
          continue;
        }

        const resolvedSearchStrategy =
          normalizeSearchStrategyValue(
            strategyConfig?.[searchStrategyParameterName],
            availableSearchStrategies,
          ) ??
          parameterDefaultStrategy ??
          'random';

        const strategyAllowedNames = await resolveAllowedNamesForModel(resolvedModelType);

        const baselineParse = parseJsonObject(strategyConfig?.baseline_hyperparameters);
        const baselineValues = baselineParse.value ?? {};
        const sanitizedBaseline =
          strategyAllowedNames.size > 0
            ? filterHyperparametersByFields(baselineValues, strategyAllowedNames)
            : cloneJson(baselineValues);

        const searchSpaceParse = parseJsonObject(strategyConfig?.search_space);
        if (searchSpaceParse.error) {
          issues.push(`${strategyLabel} has invalid search space: ${searchSpaceParse.error}`);
          continue;
        }

        const normalizedSearch = normalizeSearchSpace(searchSpaceParse.value);
        if (normalizedSearch.error) {
          issues.push(`${strategyLabel} has invalid search space: ${normalizedSearch.error}`);
          continue;
        }

        const rawSearchSpace = normalizedSearch.value ?? {};
        const sanitizedSearchSpace =
          strategyAllowedNames.size > 0
            ? filterSearchSpaceByFields(rawSearchSpace, strategyAllowedNames)
            : cloneJson(rawSearchSpace);

        if (!sanitizedSearchSpace || Object.keys(sanitizedSearchSpace).length === 0) {
          issues.push(`${strategyLabel} requires at least one hyperparameter in the search space.`);
          continue;
        }

        if (Object.values(strategy.searchSpaceFieldErrors ?? {}).some((message) => Boolean(message))) {
          issues.push(`${strategyLabel} has search space entries that need attention.`);
          continue;
        }

        const parameterCount = Object.keys(sanitizedSearchSpace).length;
        let candidateProduct: number | null = null;
        if (parameterCount > 0) {
          let product = 1;
          let overflow = false;
          Object.values(sanitizedSearchSpace).forEach((values: any) => {
            const size = Array.isArray(values) ? values.length : 0;
            product *= Math.max(size, 1);
            if (!Number.isFinite(product) || product > 1_000_000) {
              overflow = true;
            }
          });
          candidateProduct = overflow ? null : product;
        }

        const scoringRaw = strategyConfig?.[scoringMetricParameterName];
        const scoringValue = typeof scoringRaw === 'string' ? scoringRaw.trim() : '';
        const resolvedScoring = scoringValue.length ? scoringValue : null;

        const resolveIterations = () => {
          const rawIterations = strategyConfig?.[searchIterationsParameterName];
          let iterations: number | null = null;
          if (typeof rawIterations === 'number' && Number.isFinite(rawIterations)) {
            iterations = Math.max(1, Math.floor(rawIterations));
          } else if (typeof rawIterations === 'string') {
            const parsed = Number(rawIterations.trim());
            if (Number.isFinite(parsed) && parsed > 0) {
              iterations = Math.max(1, Math.floor(parsed));
            }
          }
          if (iterations === null && (resolvedSearchStrategy === 'random' || resolvedSearchStrategy === 'optuna')) {
            return searchIterationsDefault !== null ? searchIterationsDefault : null;
          }
          return iterations;
        };

        const resolveRandomState = () => {
          const rawState = strategyConfig?.[searchRandomStateParameterName];
          let value: number | null = null;
          if (typeof rawState === 'number' && Number.isFinite(rawState)) {
            value = Math.trunc(rawState);
          } else if (typeof rawState === 'string') {
            const parsed = Number(rawState.trim());
            if (Number.isFinite(parsed)) {
              value = Math.trunc(parsed);
            }
          }
          if (value === null && (resolvedSearchStrategy === 'random' || resolvedSearchStrategy === 'optuna')) {
            return searchRandomStateDefault !== null ? searchRandomStateDefault : null;
          }
          return value;
        };

        const resolvedIterations = resolveIterations();
        const resolvedRandomState = resolveRandomState();

        const metadata: Record<string, any> = {};
        if (targetColumn) {
          metadata.target_column = targetColumn;
        }
        if (problemType) {
          metadata.problem_type = problemType;
        }
        metadata.strategy_id = strategy.id;
        metadata.strategy_label = strategyLabel;
        metadata.strategy_index = index + 1;
        metadata.search_strategy = resolvedSearchStrategy;
        if (resolvedIterations !== null && (resolvedSearchStrategy === 'random' || resolvedSearchStrategy === 'optuna')) {
          metadata.max_iterations = resolvedIterations;
        }
        if (resolvedScoring) {
          metadata.scoring_metric = resolvedScoring;
        }
        if (resolvedRandomState !== null) {
          metadata.random_state = resolvedRandomState;
        }
        metadata.search_space_keys = Object.keys(sanitizedSearchSpace);
        if (parameterCount > 0) {
          metadata.search_space_parameters = parameterCount;
        }
        if (candidateProduct !== null) {
          metadata.search_space_candidates = candidateProduct;
        }
        metadata.cross_validation = {
          enabled: cvEnabled,
          strategy: cvStrategy,
          folds: cvFolds,
          shuffle: cvShuffle,
          random_state: cvRandomState,
        };

        const graphForStrategy = mergeConfigIntoGraph(strategyConfig) ?? graphPayload;
        if (!graphForStrategy) {
          issues.push(`Unable to build pipeline graph for ${strategyLabel}.`);
          continue;
        }

        const baselinePayload =
          sanitizedBaseline && Object.keys(sanitizedBaseline).length > 0 ? cloneJson(sanitizedBaseline) : undefined;
        const searchSpacePayload = cloneJson(sanitizedSearchSpace);

        const payload: HyperparameterTuningJobCreatePayload = {
          dataset_source_id: sourceId,
          pipeline_id: pipelineId,
          node_id: nodeId,
          model_type: resolvedModelType,
          model_types: [resolvedModelType],
          search_strategy: resolvedSearchStrategy,
          search_space: searchSpacePayload,
          baseline_hyperparameters: baselinePayload,
          n_iterations:
            resolvedSearchStrategy === 'random' || resolvedSearchStrategy === 'optuna'
              ? resolvedIterations ?? undefined
              : undefined,
          scoring: resolvedScoring ?? undefined,
          random_state: resolvedRandomState ?? undefined,
          cross_validation: {
            enabled: cvEnabled,
            strategy: cvStrategy,
            folds: cvFolds,
            shuffle: cvShuffle,
            random_state: cvRandomState,
          },
          metadata,
          job_metadata: metadata,
          run_tuning: true,
          graph: graphForStrategy,
          target_node_id: nodeId,
        };

        plans.push({ payload, label: strategyLabel });
      }

      return { plans, issues };
    };

    setBatchEnqueueError(null);
    const { plans, issues } = await buildStrategyPlans();
    if (!plans.length) {
      if (issues.length) {
        setBatchEnqueueError(issues.join(' '));
      }
      return;
    }

    const aggregatedJobs: HyperparameterTuningJobResponse[] = [];
    for (const { payload } of plans) {
      try {
        const result = await enqueueTuningJob(payload);
        if (Array.isArray(result?.jobs) && result.jobs.length) {
          aggregatedJobs.push(...result.jobs);
        }
      } catch (error) {
        if (error instanceof Error && error.message) {
          setBatchEnqueueError(error.message);
        } else {
          setBatchEnqueueError('Failed to enqueue one or more tuning jobs.');
        }
        return;
      }
    }

    if (aggregatedJobs.length) {
      const latestJob = aggregatedJobs[aggregatedJobs.length - 1];
      setLastCreatedJob(latestJob);
      setLastCreatedJobCount(aggregatedJobs.length);
    }
  }, [
    allowedHyperparamNames,
    cvEnabled,
    cvFolds,
    cvRandomState,
    cvShuffle,
    cvStrategy,
    enqueueTuningJob,
    graphPayload,
    mergeConfigIntoGraph,
    modelType,
    modelTypeParameterName,
    nodeId,
    onSaveDraftConfig,
    parameterDefaultStrategy,
    queryClient,
    pipelineId,
    problemType,
    scoringMetricParameterName,
    searchIterationsDefault,
    searchIterationsParameterName,
    searchRandomStateDefault,
    searchRandomStateParameterName,
    searchStrategyParameterName,
    setBatchEnqueueError,
    setHasDraftChanges,
    sourceId,
    strategies,
    targetColumn,
  ]);

  const handleRefreshJobs = useCallback(() => {
    if (shouldFetchJobs) {
      tuningJobsQuery.refetch();
    }
  }, [shouldFetchJobs, tuningJobsQuery]);

  const renderJobSummary = (job: HyperparameterTuningJobSummary) => {
    const statusLabel = STATUS_LABEL[job.status] ?? job.status;
    const modelOptionLabel = modelTypeOptions.find((option) => option.value === job.model_type)?.label;
    const modelLabel = (modelOptionLabel ?? job.model_type ?? '').trim();
    const strategyLabel = (() => {
      switch (job.search_strategy) {
        case 'grid':
          return 'Grid search';
        case 'halving':
          return 'Successive halving';
        case 'halving_random':
          return 'Successive halving (randomized)';
        case 'optuna':
          return 'Optuna search';
        default:
          return 'Random search';
      }
    })();
    const updatedLabel = job.updated_at ? formatRelativeTime(job.updated_at) : null;
    const fallbackUpdated = job.updated_at || job.created_at;
    const timestampLabel = fallbackUpdated
      ? updatedLabel ?? new Date(fallbackUpdated).toLocaleString()
      : null;
    const jobError = typeof job.error_message === 'string' ? job.error_message.trim() : '';

    const detailParts: string[] = [];
    if (modelLabel) {
      detailParts.push(`Model ${modelLabel}`);
    }
    detailParts.push(strategyLabel);
    if (Array.isArray(job.results) && job.results.length > 0) {
      detailParts.push(`${job.results.length} candidates`);
    }
    const bestScore = typeof job.best_score === 'number' && Number.isFinite(job.best_score)
      ? job.best_score
      : typeof job.metrics?.best_score === 'number'
        ? job.metrics.best_score
        : null;
    if (bestScore !== null) {
      detailParts.push(`Best score ${formatMetricValue(bestScore)}`);
    }
    const scoring = String(job.metrics?.scoring ?? job.metrics?.scorer ?? job.metadata?.scoring_metric ?? '').trim();
    if (scoring) {
      detailParts.push(`Scoring ${scoring}`);
    }
    if (job.status === 'failed' && jobError) {
      const normalizedError = jobError.replace(/\s+/g, ' ');
      detailParts.push(`Error: ${normalizedError}`);
    }
    if (timestampLabel) {
      detailParts.push(`Updated ${timestampLabel}`);
    }

    const isRunning = job.status === 'running' || job.status === 'queued';
    const progress = job.progress ?? 0;
    const currentStep = job.current_step;

    return (
      <li key={job.id} style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span>
              <strong>Run {job.run_number}</strong> — {statusLabel}
            </span>
            {isRunning && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleCancel(job.id);
                }}
                disabled={cancelMutation.isPending}
                title="Cancel this job (may not be immediate on Windows)"
                style={{
                  padding: '2px 8px',
                  fontSize: '0.75rem',
                  fontWeight: 500,
                  color: '#b91c1c',
                  background: '#fee2e2',
                  border: '1px solid #fca5a5',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            )}
          </div>
          {isRunning && (
            <span style={{ fontSize: '0.85em', color: '#666' }}>
              {progress}% {currentStep ? `— ${currentStep}` : ''}
            </span>
          )}
        </div>
        {isRunning && (
          <div
            style={{
              width: '100%',
              height: '4px',
              backgroundColor: '#eee',
              borderRadius: '2px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progress}%`,
                height: '100%',
                background: 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        )}
        <div style={{ fontSize: '0.9em', color: '#666' }}>
          {detailParts.length ? detailParts.join(' • ') : ''}
        </div>
      </li>
    );
  };

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Hyperparameter tuning jobs</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleAddStrategy}
          >
            Add strategy
          </button>
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleRefreshJobs}
            disabled={!shouldFetchJobs || isJobsLoading}
          >
            {isJobsLoading ? 'Refreshing…' : 'Refresh jobs'}
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Launch background tuning jobs that evaluate candidate hyperparameters via cross-validation. Jobs are
        versioned per pipeline and surface the best configuration when finished.
      </p>

      {pipelineId && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Pipeline ID: <code>{pipelineId}</code>
        </p>
      )}

      {hasDraftChanges && (
        <p
          className="canvas-modal__note canvas-modal__note--warning"
          style={{
            background: 'rgba(251, 146, 60, 0.1)',
            borderLeft: '3px solid rgba(251, 146, 60, 0.8)',
            padding: '0.75rem 1rem',
            margin: '0.75rem 0',
            display: 'flex',
            alignItems: 'start',
            gap: '0.5rem'
          }}
        >
          <AlertTriangle size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
          <span>
            <strong>Unsaved Configuration Changes</strong>
            <br />
            Enqueuing a tuning job saves the current configuration so results align with this pipeline snapshot.
          </span>
        </p>
      )}

      {prerequisites.map((note, index) => (
        <p key={`tuning-prereq-${index}`} className="canvas-modal__note canvas-modal__note--warning">
          {note}
        </p>
      ))}

      {createJobError instanceof Error && (
        <p className="canvas-modal__note canvas-modal__note--error">{createJobError.message}</p>
      )}

      {batchEnqueueError && (
        <p className="canvas-modal__note canvas-modal__note--error">{batchEnqueueError}</p>
      )}

      {jobsError && !(createJobError instanceof Error) && (
        <p className="canvas-modal__note canvas-modal__note--error">
          {jobsError.message || 'Unable to load tuning jobs.'}
        </p>
      )}

      {lastCreatedJob && (
        <p className="canvas-modal__note canvas-modal__note--info">
          {lastCreatedJobCount > 1 ? (
            <span>
              Queued {lastCreatedJobCount} tuning jobs. Latest job {lastCreatedJob.id} (run {lastCreatedJob.run_number}).
            </span>
          ) : (
            <span>
              Tuning job {lastCreatedJob.id} queued (run {lastCreatedJob.run_number}).
            </span>
          )}
        </p>
      )}

      {hasSearchSpaceEntries && searchSpaceDimension.parameterCount > 0 && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Search space covers {searchSpaceDimension.parameterCount} hyperparameter{searchSpaceDimension.parameterCount === 1 ? '' : 's'}
          {searchSpaceDimension.candidateProduct !== null
            ? ` • up to ${searchSpaceDimension.candidateProduct.toLocaleString()} combinations`
            : ''}
          .
        </p>
      )}

      <div className="canvas-imputer__list" style={{ marginTop: '1.25rem' }}>
        {strategies.length === 0 && (
          <p
            style={{
              margin: '0',
              padding: '1rem 1.25rem',
              border: '1px dashed rgba(148, 163, 184, 0.4)',
              borderRadius: '6px',
              color: 'rgba(148, 163, 184, 0.85)',
              background: 'rgba(15, 23, 42, 0.35)',
            }}
          >
            No tuning strategies configured yet. Use "Add strategy" to define your search plan.
          </p>
        )}
        {strategies.map((strategy, index) => {
          const summary = strategySummaries[index];
          const isActive = Boolean(activeStrategy && activeStrategy.id === strategy.id);
          const isExpanded = expandedStrategyIds.has(strategy.id);
          const summaryParts: string[] = [];
          if (summary?.modelLabel) {
            summaryParts.push(summary.modelLabel);
          }
          if (summary?.searchStrategyLabel) {
            summaryParts.push(summary.searchStrategyLabel);
          }
          if (summary?.scoringLabel) {
            summaryParts.push(`Scoring ${summary.scoringLabel}`);
          }
          if (summary?.parameterCount) {
            const candidateLabel = summary.candidateCount
              ? `${summary.parameterCount} param${summary.parameterCount === 1 ? '' : 's'} • ${summary.candidateCount} candidate${summary.candidateCount === 1 ? '' : 's'}`
              : `${summary.parameterCount} param${summary.parameterCount === 1 ? '' : 's'}`;
            summaryParts.push(candidateLabel);
          } else if (!summaryParts.includes('Needs attention')) {
            summaryParts.push('No search values');
          }
          if (summary?.hasErrors) {
            summaryParts.push('Needs attention');
          }
          const summaryText = summaryParts.length > 0 ? summaryParts.join(' · ') : 'Configure strategy settings';

          return (
            <div key={strategy.id} className="canvas-imputer__card">
              <div className="canvas-imputer__card-header">
                <button
                  type="button"
                  className="canvas-imputer__card-toggle"
                  onClick={() => {
                    handleToggleStrategy(strategy.id);
                    if (!isActive) {
                      handleActivateStrategy(strategy.id);
                    }
                  }}
                  aria-expanded={isExpanded}
                  aria-controls={`tuning-strategy-body-${strategy.id}`}
                >
                  <span
                    className={`canvas-imputer__toggle-icon${isExpanded ? ' canvas-imputer__toggle-icon--open' : ''}`}
                    aria-hidden="true"
                  />
                  <span className="canvas-imputer__card-text">
                    <span className="canvas-imputer__card-title">{summary?.label ?? createStrategyLabel(index + 1)}</span>
                    <span className="canvas-imputer__card-summary">{summaryText}</span>
                  </span>
                </button>
                {strategies.length > 1 && (
                  <button
                    type="button"
                    className="canvas-imputer__remove"
                    onClick={() => handleRemoveStrategy(strategy.id)}
                    aria-label={`Remove ${summary?.label ?? createStrategyLabel(index + 1)}`}
                  >
                    Remove
                  </button>
                )}
              </div>
              {isExpanded && isActive && (
                <div className="canvas-imputer__card-body" id={`tuning-strategy-body-${strategy.id}`}>
                  <div className="canvas-modal__parameter-grid">
                    {filteredModelTypeParameter && renderParameterField(filteredModelTypeParameter)}
                    {searchStrategyParameter && renderParameterField(searchStrategyParameter)}
                    {(searchStrategy === 'random' || searchStrategy === 'optuna') &&
                      searchIterationsParameter &&
                      renderParameterField(searchIterationsParameter)}
                    {(searchStrategy === 'random' || searchStrategy === 'optuna') &&
                      searchRandomStateParameter &&
                      renderParameterField(searchRandomStateParameter)}
                  </div>

                  <div
                    style={{
                      marginTop: '1rem',
                      padding: '1rem 1.25rem',
                      background: 'rgba(15, 23, 42, 0.25)',
                      border: '1px solid rgba(148, 163, 184, 0.2)',
                      borderRadius: '6px',
                    }}
                  >
                    <div style={{ fontWeight: 600, color: '#e2e8f0', display: 'block' }}>{scoringLabel}</div>
                    <p style={{ margin: '0.4rem 0 0.75rem', fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
                      {scoringDescription}
                    </p>
                    <div
                      id={scoringInputId}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.4rem 0.75rem',
                        borderRadius: '999px',
                        border: '1px solid rgba(148, 163, 184, 0.35)',
                        background: 'rgba(15, 23, 42, 0.35)',
                        color: 'rgba(203, 213, 225, 0.9)',
                        fontSize: '0.85rem',
                        maxWidth: '320px',
                        marginBottom: '0.75rem',
                      }}
                    >
                      <span style={{ opacity: 0.75 }}>Selected:</span>
                      <strong style={{ color: '#e2e8f0' }}>{scoringDraft || 'Estimator default'}</strong>
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: '0.5rem',
                        marginTop: '0.75rem',
                      }}
                    >
                      {scoringSuggestions.map((suggestion) => (
                        <button
                          key={suggestion.value}
                          type="button"
                          onClick={() => handleScoringChange(suggestion.value)}
                          style={{
                            padding: '0.35rem 0.7rem',
                            fontSize: '0.8rem',
                            borderRadius: '999px',
                            border: '1px solid rgba(148, 163, 184, 0.35)',
                            background: scoringDraft === suggestion.value ? 'rgba(59, 130, 246, 0.2)' : 'rgba(15, 23, 42, 0.35)',
                            color: scoringDraft === suggestion.value ? '#bfdbfe' : 'rgba(203, 213, 225, 0.9)',
                            cursor: 'pointer',
                          }}
                        >
                          {suggestion.label}
                        </button>
                      ))}
                      <button
                        type="button"
                        onClick={() => handleScoringChange('')}
                        style={{
                          padding: '0.35rem 0.7rem',
                          fontSize: '0.8rem',
                          borderRadius: '999px',
                          border: '1px solid rgba(148, 163, 184, 0.35)',
                          background: 'rgba(15, 23, 42, 0.35)',
                          color: 'rgba(203, 213, 225, 0.9)',
                          cursor: 'pointer',
                        }}
                      >
                        Reset to default
                      </button>
                    </div>
                  </div>

                  {modelType && hyperparamFields.length > 0 && (
                    <div
                      style={{
                        marginTop: '1.25rem',
                        padding: '1.25rem',
                        background: 'rgba(15, 23, 42, 0.2)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: '6px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                        <div>
                          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#e2e8f0' }}>Baseline hyperparameter overrides</h4>
                          <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
                            Pre-fill model defaults before the search runs. Leave fields blank to keep the template values.
                          </p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
                            {baselineOverrideCount > 0
                              ? `${baselineOverrideCount} override${baselineOverrideCount === 1 ? '' : 's'} configured`
                              : 'Using template defaults'}
                          </span>
                          {baselineOverrideCount > 0 && (
                            <button
                              type="button"
                              onClick={handleResetBaselineOverrides}
                              style={{
                                padding: '0.35rem 0.7rem',
                                fontSize: '0.8rem',
                                borderRadius: '999px',
                                border: '1px solid rgba(148, 163, 184, 0.35)',
                                background: 'rgba(15, 23, 42, 0.35)',
                                color: 'rgba(203, 213, 225, 0.9)',
                                cursor: 'pointer',
                              }}
                            >
                              Reset overrides
                            </button>
                          )}
                        </div>
                      </div>
                      {baselineError && (
                        <p className="canvas-modal__note canvas-modal__note--error" style={{ marginTop: '0.75rem' }}>
                          {baselineError}
                        </p>
                      )}
                      <div
                        style={{
                          marginTop: '1rem',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                          gap: '1rem',
                        }}
                      >
                        {hyperparamFields.map(renderBaselineField)}
                      </div>
                    </div>
                  )}

                  {modelType && hyperparamFields.length > 0 && (
                    <div
                      style={{
                        marginTop: '1.25rem',
                        padding: '1.25rem',
                        background: 'rgba(15, 23, 42, 0.2)',
                        border: '1px solid rgba(148, 163, 184, 0.2)',
                        borderRadius: '6px',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                        <div>
                          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: '#e2e8f0' }}>Search space</h4>
                          <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
                            Provide candidate values for each hyperparameter. Separate entries with commas or new lines.
                          </p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem' }}>
                          <span style={{ fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
                            {searchOverrideCount > 0
                              ? `${searchOverrideCount} parameter${searchOverrideCount === 1 ? '' : 's'} targeted`
                              : 'No parameters queued for tuning yet'}
                          </span>
                          <button
                            type="button"
                            onClick={handleClearSearchSpace}
                            disabled={searchOverrideCount === 0}
                            style={{
                              padding: '0.35rem 0.7rem',
                              fontSize: '0.8rem',
                              borderRadius: '999px',
                              border: '1px solid rgba(148, 163, 184, 0.35)',
                              background: searchOverrideCount === 0 ? 'rgba(15, 23, 42, 0.2)' : 'rgba(15, 23, 42, 0.35)',
                              color: searchOverrideCount === 0 ? 'rgba(148, 163, 184, 0.5)' : 'rgba(203, 213, 225, 0.9)',
                              cursor: searchOverrideCount === 0 ? 'not-allowed' : 'pointer',
                            }}
                          >
                            Clear search space
                          </button>
                        </div>
                      </div>
                      {searchSpaceError && (
                        <p className="canvas-modal__note canvas-modal__note--error" style={{ marginTop: '0.75rem' }}>
                          {searchSpaceError}
                        </p>
                      )}
                      {hasSearchSpaceFieldErrors && (
                        <p className="canvas-modal__note canvas-modal__note--warning" style={{ marginTop: '0.75rem' }}>
                          Fix the highlighted search entries before launching tuning jobs.
                        </p>
                      )}
                      <div
                        style={{
                          marginTop: '1rem',
                          display: 'grid',
                          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                          gap: '1rem',
                        }}
                      >
                        {hyperparamFields.map(renderSearchSpaceField)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {cvEnabled && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Cross-validation evaluates each candidate with {cvFolds} folds before reporting aggregate metrics.
        </p>
      )}

      {scalingWarning && (
        <div
          className="canvas-modal__note canvas-modal__note--warning"
          style={{
            margin: '0.75rem 0',
            borderLeft: '3px solid rgba(251, 191, 36, 0.9)',
            background: 'rgba(251, 191, 36, 0.12)',
            padding: '0.75rem 1rem',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'start' }}>
              <AlertTriangle size={16} style={{ marginTop: '2px', flexShrink: 0 }} />
              <div>
                <strong>{scalingWarning.headline}</strong>
                <p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem' }}>{scalingWarning.summary}</p>
              </div>
            </div>
            <button
              type="button"
              className="btn btn-outline-secondary"
              onClick={() => setShowScalingDetails((previous) => !previous)}
              style={{ whiteSpace: 'nowrap' }}
            >
              {showScalingDetails ? 'Hide tips' : 'Show tips'}
            </button>
          </div>
          {showScalingDetails && (
            <ul
              style={{
                margin: '0.75rem 0 0 1rem',
                padding: 0,
                listStyle: 'disc',
                fontSize: '0.85rem',
              }}
            >
              {scalingWarning.details.map((tip, index) => (
                <li key={`tuning-scaling-tip-${index}`} style={{ marginBottom: '0.35rem' }}>
                  {tip}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="canvas-modal__actions">
        <button type="button" className="btn btn-primary" onClick={handleEnqueueTuning} disabled={isActionDisabled}>
          {isCreatingJob ? 'Queuing…' : 'Launch tuning job'}
        </button>
      </div>

      {isJobsLoading && <p className="canvas-modal__note canvas-modal__note--muted">Loading recent jobs…</p>}
      {!isJobsLoading && tuningJobs.length === 0 && (
        <p className="canvas-modal__note canvas-modal__note--muted">No tuning jobs found for this node.</p>
      )}
      {tuningJobs.length > 0 && <ul className="canvas-modal__note-list">{tuningJobs.map(renderJobSummary)}</ul>}
    </section>
  );
};
