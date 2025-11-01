import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type {
  FeatureGraph,
  FeatureNodeParameter,
  FeatureNodeParameterOption,
  HyperparameterTuningJobListResponse,
  HyperparameterTuningJobResponse,
  HyperparameterTuningJobSummary,
  ModelHyperparameterField,
  ModelHyperparametersResponse,
} from '../../../../api';
import {
  createHyperparameterTuningJob,
  fetchHyperparameterTuningJobs,
  fetchModelHyperparameters,
  generatePipelineId,
  type FetchHyperparameterTuningJobsOptions,
} from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';
import type { TrainModelDraftConfig } from './TrainModelDraftSection';
import type { TrainModelCVConfig } from '../../hooks/useModelingConfiguration';

const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Succeeded',
  failed: 'Failed',
  cancelled: 'Cancelled',
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
  onSaveDraftConfig?: () => void;
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

    if (field.type === 'number') {
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
}) => {
  const queryClient = useQueryClient();
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [lastCreatedJob, setLastCreatedJob] = useState<HyperparameterTuningJobResponse | null>(null);
  const [pipelineIdFromSavedConfig, setPipelineIdFromSavedConfig] = useState<string | null>(null);
  const [hasDraftChanges, setHasDraftChanges] = useState(false);

  const targetColumn = (config?.targetColumn ?? '').trim();
  const problemType = config?.problemType === 'regression' ? 'regression' : 'classification';
  const modelType = (runtimeConfig?.modelType ?? '').trim();

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

  const rawSearchStrategy = String(draftConfigState?.search_strategy ?? '').trim().toLowerCase();
  const searchStrategy: 'grid' | 'random' = rawSearchStrategy === 'grid' ? 'grid' : 'random';

  const rawIterations = draftConfigState?.search_iterations;
  let searchIterations: number | null = null;
  if (typeof rawIterations === 'number' && Number.isFinite(rawIterations)) {
    searchIterations = Math.max(1, Math.floor(rawIterations));
  } else if (typeof rawIterations === 'string') {
    const parsed = Number(rawIterations.trim());
    if (Number.isFinite(parsed) && parsed > 0) {
      searchIterations = Math.max(1, Math.floor(parsed));
    }
  }

  const rawSearchRandomState = draftConfigState?.search_random_state;
  let searchRandomState: number | null = null;
  if (typeof rawSearchRandomState === 'number' && Number.isFinite(rawSearchRandomState)) {
    searchRandomState = Math.trunc(rawSearchRandomState);
  } else if (typeof rawSearchRandomState === 'string') {
    const parsed = Number(rawSearchRandomState.trim());
    if (Number.isFinite(parsed)) {
      searchRandomState = Math.trunc(parsed);
    }
  }

  const rawScoringMetric = typeof draftConfigState?.scoring_metric === 'string'
    ? draftConfigState.scoring_metric.trim()
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

  useEffect(() => {
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
              <option key={option.value} value={option.value}>
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

  const graphWithDraftConfig = useMemo<FeatureGraph | null>(() => {
    if (!graphPayload) {
      return null;
    }
    if (!draftConfigState || Object.keys(draftConfigState).length === 0) {
      return graphPayload;
    }
    const clonedGraph = cloneJson(graphPayload);
    if (!clonedGraph || !Array.isArray(clonedGraph.nodes)) {
      return clonedGraph;
    }
    clonedGraph.nodes = clonedGraph.nodes.map((node: any) => {
      if (!node || node.id !== nodeId) {
        return node;
      }
      const existingData = node?.data ?? {};
      const existingConfig = existingData?.config ?? {};
      const mergedConfig = { ...existingConfig, ...draftConfigState };
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
  }, [draftConfigState, graphPayload, nodeId]);

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
      return hasActiveJob ? 5000 : false;
    },
  });

  const {
    mutateAsync: enqueueTuningJob,
    isPending: isCreatingJob,
    error: createJobError,
  } = useMutation({
    mutationFn: createHyperparameterTuningJob,
    onSuccess: (job) => {
      setLastCreatedJob(job);
      queryClient.invalidateQueries({ queryKey: tuningJobsQueryKey });
    },
  });

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
    if (validationConnectionStatus !== true) {
      notes.push('Connect the validation split to this node to enable tuning.');
    }
    if (!modelType) {
      notes.push('Choose a model type before enqueuing tuning jobs.');
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
    if (searchStrategy === 'random' && !searchIterations) {
      notes.push('Provide the maximum iterations for random search.');
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
  ]);

  const tuningJobs = tuningJobsQuery.data?.jobs ?? [];
  const isJobsLoading = tuningJobsQuery.isLoading || tuningJobsQuery.isFetching;
  const jobsError = tuningJobsQuery.error as Error | null;

  const isActionDisabled =
    prerequisites.length > 0 ||
    hasSearchSpaceFieldErrors ||
    !pipelineId ||
    !sourceId ||
    !graphPayload ||
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
    if (!pipelineId || !sourceId || !modelType || !hasSearchSpaceEntries) {
      return;
    }

    if (onSaveDraftConfig) {
      try {
        const maybeResult: void | Promise<unknown> = (
          onSaveDraftConfig as unknown as () => void | Promise<unknown>
        )();
        setHasDraftChanges(false);
        if (maybeResult && typeof (maybeResult as Promise<unknown>).then === 'function') {
          await (maybeResult as Promise<unknown>);
        }
      } catch (error) {
        // ignore save errors here; enqueue API will surface problems if relevant
      }
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const graphForTuning = graphWithDraftConfig ?? graphPayload;
    if (!graphForTuning) {
      return;
    }

  const metadata: Record<string, any> = {};
    if (targetColumn) {
      metadata.target_column = targetColumn;
    }
    if (problemType) {
      metadata.problem_type = problemType;
    }
    metadata.search_strategy = searchStrategy;
    if (searchIterations) {
      metadata.max_iterations = searchIterations;
    }
    if (scoringMetric) {
      metadata.scoring_metric = scoringMetric;
    }
    if (hasSearchSpaceEntries) {
      metadata.search_space_keys = Object.keys(filteredSearchOverrides);
    }
    const { parameterCount, candidateProduct } = searchSpaceDimension;
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

    const metadataPayload = Object.keys(metadata).length ? metadata : undefined;

    const enhancedGraph = cloneJson(graphForTuning);
    if (enhancedGraph && Array.isArray(enhancedGraph.nodes)) {
      enhancedGraph.nodes = enhancedGraph.nodes.map((node: any) => {
        if (!node || node.id !== nodeId) {
          return node;
        }
        const existingData = node?.data ?? {};
        const existingConfig = existingData?.config ?? {};
        const mergedConfig = draftConfigState
          ? { ...existingConfig, ...draftConfigState }
          : existingConfig;
        return {
          ...node,
          data: {
            ...existingData,
            config: mergedConfig,
            isConfigured: true,
          },
        };
      });
    }

    const filteredBaselinePayload = Object.keys(filteredBaselineOverrides).length
      ? cloneJson(filteredBaselineOverrides)
      : {};
    const filteredSearchSpacePayload = cloneJson(filteredSearchOverrides);

    const payload = {
      dataset_source_id: sourceId,
      pipeline_id: pipelineId,
      node_id: nodeId,
      model_type: modelType,
      search_strategy: searchStrategy,
      search_space: filteredSearchSpacePayload,
      baseline_hyperparameters: filteredBaselinePayload,
      n_iterations: searchStrategy === 'random' ? searchIterations ?? undefined : undefined,
      scoring: scoringMetric ?? undefined,
      random_state: searchRandomState ?? undefined,
      cross_validation: {
        enabled: cvEnabled,
        strategy: cvStrategy,
        folds: cvFolds,
        shuffle: cvShuffle,
        random_state: cvRandomState,
      },
      metadata: metadataPayload,
      job_metadata: metadataPayload,
      run_tuning: true,
      graph: enhancedGraph ?? cloneJson(graphForTuning),
      target_node_id: nodeId,
    };

    try {
      await enqueueTuningJob(payload);
    } catch (error) {
      // surfaced via createJobError
    }
  }, [
    cvEnabled,
    cvFolds,
    cvRandomState,
    cvShuffle,
    cvStrategy,
    draftConfigState,
    enqueueTuningJob,
    filteredBaselineOverrides,
    filteredSearchOverrides,
    graphPayload,
    graphWithDraftConfig,
    hasDraftChanges,
    hasSearchSpaceEntries,
    modelType,
    nodeId,
    onSaveDraftConfig,
    pipelineId,
    problemType,
    scoringMetric,
    searchIterations,
    searchRandomState,
    searchSpaceDimension,
    searchStrategy,
    setHasDraftChanges,
    sourceId,
    targetColumn,
  ]);

  const handleRefreshJobs = useCallback(() => {
    if (shouldFetchJobs) {
      tuningJobsQuery.refetch();
    }
  }, [shouldFetchJobs, tuningJobsQuery]);

  const renderJobSummary = (job: HyperparameterTuningJobSummary) => {
    const statusLabel = STATUS_LABEL[job.status] ?? job.status;
    const strategyLabel = job.search_strategy === 'grid' ? 'Grid search' : 'Random search';
    const updatedLabel = job.updated_at ? formatRelativeTime(job.updated_at) : null;
    const fallbackUpdated = job.updated_at || job.created_at;
    const timestampLabel = fallbackUpdated
      ? updatedLabel ?? new Date(fallbackUpdated).toLocaleString()
      : null;

    const detailParts: string[] = [];
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
    if (timestampLabel) {
      detailParts.push(`Updated ${timestampLabel}`);
    }

    return (
      <li key={job.id}>
        <strong>Run {job.run_number}</strong> — {statusLabel}
        {detailParts.length ? ` • ${detailParts.join(' • ')}` : ''}
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
          }}
        >
          <strong>⚠️ Unsaved Configuration Changes</strong>
          <br />
          Enqueuing a tuning job saves the current configuration so results align with this pipeline snapshot.
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

      {jobsError && !(createJobError instanceof Error) && (
        <p className="canvas-modal__note canvas-modal__note--error">
          {jobsError.message || 'Unable to load tuning jobs.'}
        </p>
      )}

      {lastCreatedJob && (
        <p className="canvas-modal__note canvas-modal__note--info">
          Tuning job {lastCreatedJob.id} queued (run {lastCreatedJob.run_number}).
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

      <div className="canvas-modal__parameter-grid">
        {filteredModelTypeParameter && renderParameterField(filteredModelTypeParameter)}
        {searchStrategyParameter && renderParameterField(searchStrategyParameter)}
        {searchStrategy === 'random' && searchIterationsParameter && renderParameterField(searchIterationsParameter)}
        {searchStrategy === 'random' && searchRandomStateParameter && renderParameterField(searchRandomStateParameter)}
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

      {cvEnabled && (
        <p className="canvas-modal__note canvas-modal__note--muted">
          Cross-validation evaluates each candidate with {cvFolds} folds before reporting aggregate metrics.
        </p>
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
