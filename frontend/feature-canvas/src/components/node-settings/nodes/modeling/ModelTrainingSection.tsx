import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type {
	FeatureGraph,
	FeatureNodeParameter,
	FeatureNodeParameterOption,
	TrainingJobListResponse,
	TrainingJobResponse,
	TrainingJobSummary,
	ModelHyperparameterField,
	ModelHyperparametersResponse,
	HyperparameterTuningJobListResponse,
	HyperparameterTuningJobSummary,
	BestHyperparametersResponse,
} from '../../../../api';
import {
	createTrainingJob,
	fetchTrainingJobs,
	generatePipelineId,
	fetchModelHyperparameters,
	type FetchTrainingJobsOptions,
	fetchHyperparameterTuningJobs,
	type FetchHyperparameterTuningJobsOptions,
	fetchBestHyperparameters,
} from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';
import type { TrainModelDraftConfig } from './TrainModelDraftSection';
import type { TrainModelCVConfig } from '../../hooks';
import { BestHyperparamsModal, type HyperparamPreset } from './BestHyperparamsModal';
import { useScalingWarning, detectScalingConvergenceFromJob, hasScalingConvergenceMessage } from './useScalingWarning';
import { TrainingJobHistory } from './TrainingJobHistory';
import { HyperparameterControls } from './HyperparameterControls';
import {
	filterHyperparametersByFields,
	valuesEqual,
	sanitizeHyperparametersForPayload,
	cloneJson,
	pickPrimaryMetric,
	formatMetricValue,
} from './modelingUtils';

export type TrainModelRuntimeConfig = {
	modelType: string | null;
	hyperparameters: Record<string, any> | null;
	hyperparametersError: string | null;
};

type ModelTrainingSectionProps = {
	nodeId: string;
	sourceId?: string | null;
	graph: FeatureGraph | null;
	config: TrainModelDraftConfig | null;
	runtimeConfig: TrainModelRuntimeConfig | null;
	cvConfig: TrainModelCVConfig;
	draftConfigState: Record<string, any> | null;
	renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
	modelTypeParameter: FeatureNodeParameter | null;
	modelTypeOptions: FeatureNodeParameterOption[];
	hyperparametersParameter: FeatureNodeParameter | null;
	cvEnabledParameter: FeatureNodeParameter | null;
	cvStrategyParameter: FeatureNodeParameter | null;
	cvFoldsParameter: FeatureNodeParameter | null;
	cvShuffleParameter: FeatureNodeParameter | null;
	cvRandomStateParameter: FeatureNodeParameter | null;
	cvRefitStrategyParameter: FeatureNodeParameter | null;
	onSaveDraftConfig?: (options?: { closeModal?: boolean }) => void | Promise<void>;
};

export const ModelTrainingSection: React.FC<ModelTrainingSectionProps> = ({
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
	hyperparametersParameter,
	cvEnabledParameter,
	cvStrategyParameter,
	cvFoldsParameter,
	cvShuffleParameter,
	cvRandomStateParameter,
	cvRefitStrategyParameter,
	onSaveDraftConfig,
}) => {
	const queryClient = useQueryClient();
	const [pipelineId, setPipelineId] = useState<string | null>(null);
	const [pipelineError, setPipelineError] = useState<string | null>(null);
	const [lastCreatedJob, setLastCreatedJob] = useState<TrainingJobResponse | null>(null);
	const [lastCreatedJobCount, setLastCreatedJobCount] = useState<number>(0);
	const [hyperparamValues, setHyperparamValues] = useState<Record<string, any>>({});
	const [showAdvanced, setShowAdvanced] = useState(false);
	const advancedInitializedRef = useRef(false);
	const [pipelineIdFromSavedConfig, setPipelineIdFromSavedConfig] = useState<string | null>(null);
	const [hasDraftChanges, setHasDraftChanges] = useState(false);
	const [isPresetModalOpen, setPresetModalOpen] = useState(false);
	const [lastAppliedPresetId, setLastAppliedPresetId] = useState<string | null>(null);
	const [applyStatus, setApplyStatus] = useState<{ message: string; tone: 'info' | 'warning' | 'success' } | null>(null);
	const [showScalingDetails, setShowScalingDetails] = useState(false);

	const targetColumn = (config?.targetColumn ?? '').trim();
	const problemType = config?.problemType === 'regression' ? 'regression' : 'classification';
	const modelType = (runtimeConfig?.modelType ?? '').trim();
	const hyperparameters = runtimeConfig?.hyperparameters ?? null;
	const hyperparametersError = runtimeConfig?.hyperparametersError ?? null;
	const cvEnabled = Boolean(cvConfig?.enabled);
	const cvStrategy = cvConfig?.strategy ?? 'auto';
	const cvFolds = typeof cvConfig?.folds === 'number' ? cvConfig.folds : 0;
	const cvShuffle = Boolean(cvConfig?.shuffle);
	const cvRandomState = cvConfig?.randomState ?? null;
	const cvRefitStrategy = cvConfig?.refitStrategy ?? 'train_plus_validation';

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

	// Fetch hyperparameter schema for the selected model
	const hyperparamQuery = useQuery<ModelHyperparametersResponse, Error>({
		queryKey: ['model-hyperparameters', modelType],
		queryFn: () => fetchModelHyperparameters(modelType),
		enabled: Boolean(modelType),
		staleTime: 5 * 60 * 1000, // Cache for 5 minutes
	});

	// Fetch best hyperparameters from tuning for the selected model type
	const bestParamsQuery = useQuery<BestHyperparametersResponse, Error>({
		queryKey: ['best-hyperparameters', modelType, pipelineIdFromSavedConfig, sourceId],
		queryFn: () => fetchBestHyperparameters(modelType, {
			pipelineId: pipelineIdFromSavedConfig || undefined,
			datasetSourceId: sourceId || undefined,
		}),
		enabled: Boolean(modelType && sourceId),
		staleTime: 30 * 1000, // Cache for 30 seconds
		retry: 1,
	});

	const hyperparamFields = hyperparamQuery.data?.fields ?? [];
	const hyperparamDefaults = hyperparamQuery.data?.defaults ?? {};
	const allowedHyperparamNames = useMemo(() => {
		if (!hyperparamFields.length) {
			return new Set<string>();
		}
		return new Set<string>(hyperparamFields.map((field) => field.name));
	}, [hyperparamFields]);
	const hyperparamFieldMap = useMemo(() => {
		if (!hyperparamFields.length) {
			return {} as Record<string, ModelHyperparameterField>;
		}
		return hyperparamFields.reduce((acc, field) => {
			if (field?.name) {
				acc[field.name] = field;
			}
			return acc;
		}, {} as Record<string, ModelHyperparameterField>);
	}, [hyperparamFields]);
	const filteredRuntimeHyperparams = useMemo(() => {
		if (!modelType || !hyperparameters) {
			return null;
		}
		if (allowedHyperparamNames.size === 0) {
			return null;
		}
		const filtered = filterHyperparametersByFields(hyperparameters, allowedHyperparamNames);
		return Object.keys(filtered).length > 0 ? filtered : null;
	}, [allowedHyperparamNames, hyperparameters, modelType]);

	useEffect(() => {
		setHyperparamValues({});
		setShowAdvanced(false);
		advancedInitializedRef.current = false;
	}, [modelType]);

	// Hydrate hyperparameters from persisted runtime configuration
	useEffect(() => {
		if (!filteredRuntimeHyperparams) {
			return;
		}
		setHyperparamValues((prev) => {
			const prevKeys = Object.keys(prev);
			const nextKeys = Object.keys(filteredRuntimeHyperparams);
			const isSame =
				prevKeys.length === nextKeys.length && nextKeys.every((key) => Object.is(prev[key], filteredRuntimeHyperparams[key]));
			if (isSame) {
				return prev;
			}
			return filteredRuntimeHyperparams;
		});
		if (!advancedInitializedRef.current) {
			setShowAdvanced(true);
			advancedInitializedRef.current = true;
		}
	}, [filteredRuntimeHyperparams]);

	// Fall back to hyperparameter defaults when nothing is persisted
	useEffect(() => {
		if (!modelType) {
			return;
		}
		if (filteredRuntimeHyperparams) {
			return;
		}
		if (allowedHyperparamNames.size === 0) {
			return;
		}
		if (!hyperparamDefaults || Object.keys(hyperparamDefaults).length === 0) {
			return;
		}
		const filteredDefaults = filterHyperparametersByFields(hyperparamDefaults, allowedHyperparamNames);
		if (Object.keys(filteredDefaults).length === 0) {
			return;
		}
		setHyperparamValues((prev) => {
			if (Object.keys(prev).length > 0) {
				return prev;
			}
			return filteredDefaults;
		});
	}, [allowedHyperparamNames, filteredRuntimeHyperparams, hyperparamDefaults, modelType]);

	const sanitizedHyperparameters = useMemo(() => {
		if (allowedHyperparamNames.size === 0) {
			return null;
		}
		const sanitized = sanitizeHyperparametersForPayload(
			hyperparamValues,
			allowedHyperparamNames,
			hyperparamFieldMap
		);
		return Object.keys(sanitized).length > 0 ? sanitized : null;
	}, [allowedHyperparamNames, hyperparamValues]);

	const handleToggleAdvanced = useCallback(() => {
		setShowAdvanced((prev) => {
			const next = !prev;
			if (next) {
				advancedInitializedRef.current = true;
			}
			return next;
		});
	}, []);

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

	// Compute pipeline ID from saved config (ignoring drafts)
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

	// Compute pipeline ID including draft changes
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

	// Detect if draft config changes would alter the pipeline ID
	useEffect(() => {
		if (!pipelineId || !pipelineIdFromSavedConfig) {
			setHasDraftChanges(false);
			return;
		}
		setHasDraftChanges(pipelineId !== pipelineIdFromSavedConfig);
	}, [pipelineId, pipelineIdFromSavedConfig]);

	const trainingJobsQueryKey = useMemo(() => {
		return ['feature-canvas', 'training-jobs', sourceId ?? 'none', pipelineId ?? 'pending', nodeId];
	}, [nodeId, pipelineId, sourceId]);

	const tuningJobsQueryKey = useMemo(() => {
		return ['feature-canvas', 'tuning-jobs', sourceId ?? 'none', pipelineId ?? 'pending', nodeId];
	}, [nodeId, pipelineId, sourceId]);

	const shouldFetchJobs = Boolean(sourceId && nodeId);

	const tuningJobsQuery = useQuery<HyperparameterTuningJobListResponse, Error>({
		queryKey: tuningJobsQueryKey,
		queryFn: async () => {
			const baseParams: FetchHyperparameterTuningJobsOptions = {
				datasetSourceId: sourceId || undefined,
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
			const hasActiveJob = jobList.some(
				(job: HyperparameterTuningJobSummary) => job.status === 'queued' || job.status === 'running'
			);
			return hasActiveJob ? 5000 : false;
		},
	});

		const trainingJobsQuery = useQuery<TrainingJobListResponse, Error>({
		queryKey: trainingJobsQueryKey,
		queryFn: async () => {
			const baseParams: FetchTrainingJobsOptions = {
				datasetSourceId: sourceId || undefined,
				nodeId,
				limit: 5,
			};

			if (pipelineId) {
				const scoped = await fetchTrainingJobs({ ...baseParams, pipelineId });
				if ((scoped?.jobs?.length ?? 0) > 0) {
					return scoped;
				}
			}

			return fetchTrainingJobs(baseParams);
		},
		enabled: shouldFetchJobs,
		retry: (failureCount, error) => {
			// Don't retry on authentication errors
			if (error.message.includes('Sign in')) {
				return false;
			}
			return failureCount < 2;
		},
		refetchInterval: (query) => {
			const currentData = query.state.data as TrainingJobListResponse | undefined;
			const jobList = currentData?.jobs ?? [];
			const hasActiveJob = jobList.some(
				(job: TrainingJobSummary) => job.status === 'queued' || job.status === 'running'
			);
			return hasActiveJob ? 5000 : false;
		},
	});

	const {
		mutateAsync: enqueueTrainingJob,
		isPending: isCreatingJob,
		error: createJobError,
	} = useMutation({
		mutationFn: createTrainingJob,
		onSuccess: (result) => {
			const jobCount = Array.isArray(result.jobs) ? result.jobs.length : 0;
			const firstJob = jobCount > 0 ? result.jobs[0] : null;
			setLastCreatedJob(firstJob);
			setLastCreatedJobCount(jobCount);
			queryClient.invalidateQueries({ queryKey: trainingJobsQueryKey });
		},
	});

	const tuningJobsError = tuningJobsQuery.error as Error | null;
	const isTuningJobsLoading = tuningJobsQuery.isLoading || tuningJobsQuery.isFetching;
	const tuningJobs: HyperparameterTuningJobSummary[] = tuningJobsQuery.data?.jobs ?? [];

	const bestParamsData = bestParamsQuery.data;

	const buildPresetParams = useCallback(
		(rawParams: Record<string, any> | null | undefined) => {
			if (!rawParams) {
				return null;
			}
			const rawKeys = Object.keys(rawParams);
			if (rawKeys.length === 0) {
				return null;
			}
			if (allowedHyperparamNames.size === 0) {
				return { params: rawParams, hasUnmapped: false };
			}
			const filtered = filterHyperparametersByFields(rawParams, allowedHyperparamNames);
			const filteredKeys = Object.keys(filtered);
			if (filteredKeys.length > 0) {
				return {
					params: filtered,
					hasUnmapped: filteredKeys.length !== rawKeys.length,
				};
			}
			return {
				params: rawParams,
				hasUnmapped: true,
			};
		},
		[allowedHyperparamNames],
	);

	const succeededTuningJobs = useMemo(() => {
		if (!tuningJobs.length) {
			return [] as HyperparameterTuningJobSummary[];
		}
		return tuningJobs.filter((job) => job.status === 'succeeded');
	}, [tuningJobs]);

	const hyperparamPresets = useMemo<HyperparamPreset[]>(() => {
		if (!modelType) {
			return [];
		}
		const presets: HyperparamPreset[] = [];
		const seenJobIds = new Set<string>();
		const normalizedTarget = targetColumn ? targetColumn.trim().toLowerCase() : '';
		const matchesTarget = (candidateTarget?: string | null) => {
			if (!normalizedTarget) {
				return true;
			}
			if (!candidateTarget) {
				return true;
			}
			return candidateTarget.trim().toLowerCase() === normalizedTarget;
		};

		if (bestParamsData?.available && bestParamsData.best_params && (!bestParamsData.model_type || bestParamsData.model_type === modelType)) {
			const paramInfo = buildPresetParams(bestParamsData.best_params);
			if (paramInfo) {
				const jobId = bestParamsData.job_id ?? null;
				if (!jobId || !seenJobIds.has(jobId)) {
					const labelSegments: string[] = ['Most recent tuned run'];
					if (bestParamsData.run_number) {
						labelSegments.push(`#${bestParamsData.run_number}`);
					}
					const presetId = `best-api-${jobId ?? modelType}`;
					presets.push({
						id: presetId,
						label: labelSegments.join(' '),
						source: 'best-api',
						modelType: bestParamsData.model_type ?? modelType,
						params: paramInfo.params,
						jobId,
						nodeId: bestParamsData.node_id ?? null,
						runNumber: bestParamsData.run_number ?? null,
						score: bestParamsData.best_score ?? null,
						scoring: bestParamsData.scoring ?? null,
						finishedAt: bestParamsData.finished_at ?? null,
						searchStrategy: bestParamsData.search_strategy ?? null,
						nIterations: bestParamsData.n_iterations ?? null,
						targetColumn: targetColumn || null,
						pipelineId: bestParamsData.pipeline_id ?? null,
						datasetSourceId: sourceId ?? null,
						description: paramInfo.hasUnmapped
							? 'Includes parameters outside the current template. They will be stored as custom overrides.'
							: null,
					});
					if (jobId) {
						seenJobIds.add(jobId);
					}
				}
			}
		}

		succeededTuningJobs.forEach((job) => {
			if (!job.best_params) {
				return;
			}
			if (job.model_type !== modelType) {
				return;
			}
			if (sourceId && job.dataset_source_id && job.dataset_source_id !== sourceId) {
				return;
			}
			if (seenJobIds.has(job.id)) {
				return;
			}
			const paramInfo = buildPresetParams(job.best_params);
			if (!paramInfo) {
				return;
			}
			const metadataTarget = typeof job.metadata?.target_column === 'string'
				? job.metadata.target_column
				: typeof job.metadata?.targetColumn === 'string'
				? job.metadata.targetColumn
				: null;
			if (!matchesTarget(metadataTarget)) {
				return;
			}
			const presetId = `job-${job.id}`;
			const label = (() => {
				const rawLabel = typeof job.metadata?.label === 'string' ? job.metadata.label.trim() : '';
				if (rawLabel) {
					return rawLabel;
				}
				if (job.run_number) {
					return `Run ${job.run_number}`;
				}
				return `Job ${job.id.slice(0, 6)}`;
			})();
			const scoringMetric = typeof job.metadata?.scoring === 'string'
				? job.metadata.scoring
				: typeof job.metrics?.scoring === 'string'
				? job.metrics.scoring
				: null;
			const iterationCount = typeof job.metadata?.n_iterations === 'number'
				? job.metadata.n_iterations
				: null;
			presets.push({
				id: presetId,
				label,
				source: 'tuning-job',
				modelType: job.model_type,
				params: paramInfo.params,
				jobId: job.id,
				nodeId: job.node_id,
				runNumber: job.run_number ?? null,
				score: job.best_score ?? null,
				scoring: scoringMetric,
				finishedAt: job.updated_at ?? job.created_at ?? null,
				searchStrategy: job.search_strategy ?? null,
				nIterations: iterationCount,
				targetColumn: metadataTarget ?? (targetColumn || null),
				pipelineId: job.pipeline_id ?? null,
				datasetSourceId: job.dataset_source_id ?? null,
				description: paramInfo.hasUnmapped
					? 'Some tuned parameters fall outside the current template and will be saved as custom overrides.'
					: null,
			});
			seenJobIds.add(job.id);
		});

		presets.sort((a, b) => {
			const aTime = a.finishedAt ? Date.parse(a.finishedAt) : 0;
			const bTime = b.finishedAt ? Date.parse(b.finishedAt) : 0;
			if (aTime !== bTime) {
				return bTime - aTime;
			}
			return (b.runNumber ?? 0) - (a.runNumber ?? 0);
		});

		return presets;
	}, [
		bestParamsData,
		buildPresetParams,
		modelType,
		sourceId,
		targetColumn,
		succeededTuningJobs,
	]);

	const primaryPreset = hyperparamPresets[0] ?? null;
	const hasPresetOptions = hyperparamPresets.length > 0;
	const applyButtonDisabled = !primaryPreset;
	const browsePresetsDisabled = !hasPresetOptions;

	const latestTuningJob = useMemo<HyperparameterTuningJobSummary | null>(() => {
		if (succeededTuningJobs.length) {
			return succeededTuningJobs
				.slice()
				.sort((a, b) => {
					const aTime = Date.parse(a.updated_at || a.created_at || '0');
					const bTime = Date.parse(b.updated_at || b.created_at || '0');
					return bTime - aTime;
				})[0];
		}
		if (tuningJobs.length) {
			return tuningJobs
				.slice()
				.sort((a, b) => {
					const aTime = Date.parse(a.updated_at || a.created_at || '0');
					const bTime = Date.parse(b.updated_at || b.created_at || '0');
					return bTime - aTime;
				})[0];
		}
		return null;
	}, [succeededTuningJobs, tuningJobs]);

	const latestTuningTimestamp = latestTuningJob
		? latestTuningJob.updated_at || latestTuningJob.created_at || null
		: null;
	const latestTuningRelative = latestTuningTimestamp ? formatRelativeTime(latestTuningTimestamp) : null;
	const latestTuningRunNumber = latestTuningJob?.run_number ?? null;
	const primaryPresetRelative = primaryPreset?.finishedAt ? formatRelativeTime(primaryPreset.finishedAt) : null;
	const primaryPresetScoreLabel = primaryPreset && primaryPreset.scoring && typeof primaryPreset.score === 'number' && Number.isFinite(primaryPreset.score)
		? `${primaryPreset.scoring}: ${formatMetricValue(primaryPreset.score)}`
		: null;

	useEffect(() => {
		if (!hyperparamPresets.length) {
			if (lastAppliedPresetId) {
				setLastAppliedPresetId(null);
			}
			if (applyStatus?.tone === 'success') {
				setApplyStatus(null);
			}
			return;
		}
		if (lastAppliedPresetId && !hyperparamPresets.some((preset) => preset.id === lastAppliedPresetId)) {
			setLastAppliedPresetId(null);
			setApplyStatus(null);
		}
	}, [applyStatus, hyperparamPresets, lastAppliedPresetId]);

	const applyPreset = useCallback(
		(preset: HyperparamPreset, messageOverride?: string) => {
			if (!preset || Object.keys(preset.params ?? {}).length === 0) {
				setApplyStatus({
					message: 'Selected preset does not include any hyperparameters to apply.',
					tone: 'warning',
				});
				return;
			}
			setHyperparamValues(() => cloneJson(preset.params));
			setShowAdvanced(true);
			advancedInitializedRef.current = true;
			setLastAppliedPresetId(preset.id);
			setApplyStatus({
				message:
					messageOverride ?? `Applied tuned parameters from ${preset.label}.`,
				tone: 'success',
			});
		},
		[setHyperparamValues, setShowAdvanced, setApplyStatus, setLastAppliedPresetId],
	);

	const handleApplyBestParams = useCallback(() => {
		if (!primaryPreset) {
			const message = latestTuningJob && latestTuningJob.model_type && latestTuningJob.model_type !== modelType
				? `Latest tuning run targeted “${latestTuningJob.model_type}”. Switch the model template to reuse its parameters.`
				: 'No tuning job results available to apply.';
			setApplyStatus({
				message,
				tone: 'warning',
			});
			return;
		}
		const successMessage = primaryPreset.runNumber
			? `Applied tuned parameters from run ${primaryPreset.runNumber}.`
			: `Applied tuned parameters from ${primaryPreset.label}.`;
		applyPreset(primaryPreset, successMessage);
	}, [applyPreset, latestTuningJob, modelType, primaryPreset, setApplyStatus]);

	const applyStatusColor = useMemo(() => {
		if (!applyStatus) {
			return 'rgba(148, 163, 184, 0.75)';
		}
		switch (applyStatus.tone) {
			case 'success':
				return 'rgba(134, 239, 172, 0.9)';
			case 'warning':
				return 'rgba(253, 186, 116, 0.95)';
			default:
				return 'rgba(148, 163, 184, 0.85)';
		}
	}, [applyStatus]);

	const prerequisites = useMemo(() => {
		const notes: string[] = [];
		if (!sourceId) {
			notes.push('Select a dataset before launching training jobs.');
		}
		if (!graphPayload || !graphPayload.nodes.length) {
			notes.push('Connect this node to an upstream pipeline before training.');
		}
		if (!targetColumn) {
			notes.push('Set a target column in the Train model readiness section.');
		}
		if (!modelType) {
			notes.push('Choose a model type before enqueuing training jobs.');
		}
		if (cvEnabled && cvFolds < 2) {
			notes.push('Cross-validation requires at least 2 folds.');
		}
		if (pipelineError) {
			notes.push(pipelineError);
		}
		if (hyperparametersError) {
			notes.push(hyperparametersError);
		}
		return notes;
	}, [cvEnabled, cvFolds, graphPayload, hyperparametersError, modelType, pipelineError, sourceId, targetColumn]);

	const jobs: TrainingJobSummary[] = trainingJobsQuery.data?.jobs ?? [];
	const filteredJobs = useMemo(() => {
		if (!jobs.length) {
			return jobs;
		}
		const normalizedProblemType = problemType?.toLowerCase();
		return jobs.filter((job) => {
			if (!normalizedProblemType) {
				return true;
			}
			const jobProblemType = (() => {
				const direct = typeof job.problem_type === 'string' ? job.problem_type : null;
				const metaProblem = typeof job.metadata?.problem_type === 'string' ? job.metadata?.problem_type : null;
				const resolved = typeof job.metadata?.resolved_problem_type === 'string' ? job.metadata?.resolved_problem_type : null;
				return (direct || resolved || metaProblem || '').toLowerCase();
			})();
			if (jobProblemType) {
				return jobProblemType === normalizedProblemType;
			}
			const modelTypeValue = String(job.model_type || '').toLowerCase();
			if (normalizedProblemType === 'regression') {
				return modelTypeValue.includes('regress');
			}
			if (normalizedProblemType === 'classification') {
				return modelTypeValue.includes('classif');
			}
			return true;
		});
	}, [jobs, problemType]);

	const hasScalingConvergenceSignals = useMemo(() => {
		if (createJobError instanceof Error && hasScalingConvergenceMessage(createJobError.message)) {
			return true;
		}
		if (lastCreatedJob && detectScalingConvergenceFromJob(lastCreatedJob)) {
			return true;
		}
		if (jobs.some((job) => detectScalingConvergenceFromJob(job))) {
			return true;
		}
		if (tuningJobs.some((job) => detectScalingConvergenceFromJob(job))) {
			return true;
		}
		return false;
	}, [createJobError, jobs, lastCreatedJob, tuningJobs]);

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

	const hiddenJobCount = Math.max(jobs.length - filteredJobs.length, 0);
	const isJobsLoading = trainingJobsQuery.isLoading || trainingJobsQuery.isFetching;
	const jobsError = trainingJobsQuery.error as Error | null;

	const isActionDisabled =
		prerequisites.length > 0 || !pipelineId || !sourceId || !graphPayload || isCreatingJob;

	const handleHyperparamChange = useCallback((name: string, value: any) => {
		setHyperparamValues((prev) => ({
			...prev,
			[name]: value,
		}));
	}, []);

	const renderHyperparamField = useCallback(
		(field: ModelHyperparameterField) => {
			const currentValue = hyperparamValues[field.name] ?? field.default;
			const fieldId = `hyperparam-${nodeId}-${field.name}`;

			return (
				<div key={field.name} className="canvas-modal__parameter">
					<label htmlFor={fieldId} className="canvas-modal__parameter-label">
						{field.label}
						{field.description && (
							<span className="canvas-modal__parameter-description">{field.description}</span>
						)}
					</label>
					{field.type === 'number' && (
						<input
							id={fieldId}
							type="number"
							className="canvas-modal__parameter-input"
							value={currentValue ?? ''}
							onChange={(e) => {
								const val = e.target.value;
								if (val === '' && field.nullable) {
									handleHyperparamChange(field.name, null);
								} else {
									const num = parseFloat(val);
									if (!isNaN(num)) {
										handleHyperparamChange(field.name, num);
									}
								}
							}}
							min={field.min}
							max={field.max}
							step={field.step}
							placeholder={field.nullable ? 'Empty = default' : ''}
						/>
					)}
					{field.type === 'select' && (
						<select
							id={fieldId}
							className="canvas-modal__parameter-select"
							value={currentValue ?? ''}
							onChange={(e) => handleHyperparamChange(field.name, e.target.value)}
						>
							{field.options?.map((opt) => (
								<option key={String(opt.value)} value={String(opt.value)}>
									{opt.label}
								</option>
							))}
						</select>
					)}
					{field.type === 'boolean' && (
						<input
							id={fieldId}
							type="checkbox"
							className="canvas-modal__parameter-checkbox"
							checked={Boolean(currentValue)}
							onChange={(e) => handleHyperparamChange(field.name, e.target.checked)}
						/>
					)}
					{field.type === 'text' && (
						<input
							id={fieldId}
							type="text"
							className="canvas-modal__parameter-input"
							value={currentValue ?? ''}
							onChange={(e) => handleHyperparamChange(field.name, e.target.value)}
							placeholder={field.nullable ? 'Empty = default' : ''}
						/>
					)}
				</div>
			);
		},
		[hyperparamValues, handleHyperparamChange, nodeId]
	);

	const handleEnqueueTraining = useCallback(async () => {
		if (!pipelineId || !sourceId || !modelType) {
			return;
		}

		// Auto-save draft config before enqueueing to stabilize pipeline ID
		if (hasDraftChanges && onSaveDraftConfig) {
			try {
				const maybeResult = onSaveDraftConfig({ closeModal: false });
				if (maybeResult && typeof (maybeResult as Promise<unknown>).then === 'function') {
					await (maybeResult as Promise<unknown>);
				}
				setHasDraftChanges(false);
			} catch (error) {
				// Ignore save errors here; job enqueue will surface actionable issues.
			}
			// Small delay to allow state propagation
			await new Promise((resolve) => setTimeout(resolve, 100));
		}

		const graphForTraining = graphWithDraftConfig ?? graphPayload;
		if (!graphForTraining) {
			return;
		}

		const metadata: Record<string, any> = {};
		if (targetColumn) {
			metadata.target_column = targetColumn;
		}
		if (problemType) {
			metadata.problem_type = problemType;
		}
		metadata.cross_validation = {
			enabled: cvEnabled,
			strategy: cvStrategy,
			folds: cvFolds,
			shuffle: cvShuffle,
			random_state: cvRandomState,
			refit_strategy: cvRefitStrategy,
		};

		const metadataPayload = Object.keys(metadata).length ? metadata : undefined;

		const enhancedGraph = cloneJson(graphForTraining);
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

		const requestedModelTypes = modelType ? [modelType] : [];

		const payload = {
			dataset_source_id: sourceId,
			pipeline_id: pipelineId,
			node_id: nodeId,
			model_type: modelType || undefined,
			model_types: requestedModelTypes.length ? requestedModelTypes : undefined,
			hyperparameters:
				showAdvanced && sanitizedHyperparameters ? cloneJson(sanitizedHyperparameters) : undefined,
			metadata: metadataPayload,
			job_metadata: metadataPayload,
			run_training: true,
			graph: enhancedGraph ?? cloneJson(graphForTraining),
			target_node_id: nodeId,
		};

		try {
			await enqueueTrainingJob(payload);
		} catch (error) {
			// Error surfaced via createJobError and rendered below.
		}
	}, [cvEnabled, cvFolds, cvRandomState, cvRefitStrategy, cvShuffle, cvStrategy, draftConfigState, enqueueTrainingJob, graphPayload, graphWithDraftConfig, hasDraftChanges, modelType, nodeId, onSaveDraftConfig, pipelineId, problemType, sanitizedHyperparameters, showAdvanced, sourceId, targetColumn]);

	const handleRefreshJobs = useCallback(() => {
		if (shouldFetchJobs) {
			trainingJobsQuery.refetch();
		}
	}, [shouldFetchJobs, trainingJobsQuery]);

	return (
		<>
			<section className="canvas-modal__section">
				<div className="canvas-modal__section-header">
					<h3>Model Training</h3>
				</div>
				<p className="canvas-modal__note">
					Configure and launch training jobs.
				</p>

				{pipelineId && (
				<p className="canvas-modal__note canvas-modal__note--muted">
					Pipeline ID: <code>{pipelineId}</code>
				</p>
			)}

			{hasDraftChanges && (
				<p className="canvas-modal__note canvas-modal__note--warning" style={{ 
					background: 'rgba(251, 146, 60, 0.1)', 
					borderLeft: '3px solid rgba(251, 146, 60, 0.8)',
					padding: '0.75rem 1rem',
					margin: '0.75rem 0'
				}}>
					<strong>⚠️ Unsaved Configuration Changes</strong>
					<br />
					Your changes will create a new training pipeline. Previous jobs remain under the old configuration.
					Jobs will auto-save when you launch training.
				</p>
			)}

			{prerequisites.map((note, index) => (
				<p key={`training-prereq-${index}`} className="canvas-modal__note canvas-modal__note--warning">
					{note}
				</p>
			))}

			{createJobError instanceof Error && (
				<p className="canvas-modal__note canvas-modal__note--error">{createJobError.message}</p>
			)}

			{jobsError && !(createJobError instanceof Error) && (
				<p className="canvas-modal__note canvas-modal__note--error">
					{jobsError.message || 'Unable to load training jobs.'}
				</p>
			)}
			{tuningJobsError && (
				<p className="canvas-modal__note canvas-modal__note--error">
					{tuningJobsError.message || 'Unable to load tuning jobs.'}
				</p>
			)}

			{lastCreatedJob && (
				<p className="canvas-modal__note canvas-modal__note--info">
					{lastCreatedJobCount > 1 ? (
						<span>
							Queued {lastCreatedJobCount} training jobs. Latest job {lastCreatedJob.id} (version {lastCreatedJob.version}).
						</span>
					) : (
						<span>
							Training job {lastCreatedJob.id} queued (version {lastCreatedJob.version}).
						</span>
					)}
				</p>
			)}
			{isTuningJobsLoading && (
				<p className="canvas-modal__note canvas-modal__note--muted">Loading tuning jobs…</p>
			)}

			<div className="canvas-modal__parameter-grid">
				{filteredModelTypeParameter && renderParameterField(filteredModelTypeParameter)}
				{cvEnabledParameter && renderParameterField(cvEnabledParameter)}
				{cvEnabled && (
					<>
						{cvStrategyParameter && renderParameterField(cvStrategyParameter)}
						{cvFoldsParameter && renderParameterField(cvFoldsParameter)}
						{cvShuffleParameter && renderParameterField(cvShuffleParameter)}
						{cvRandomStateParameter && renderParameterField(cvRandomStateParameter)}
						{cvRefitStrategyParameter && renderParameterField(cvRefitStrategyParameter)}
					</>
				)}
				
				<HyperparameterControls
					modelType={modelType}
					showAdvanced={showAdvanced}
					hyperparamFields={hyperparamFields}
					hyperparamValues={hyperparamValues}
					onHyperparamChange={handleHyperparamChange}
					onToggleAdvanced={handleToggleAdvanced}
					isLoading={hyperparamQuery.isLoading}
					error={hyperparamQuery.error}
					primaryPreset={primaryPreset}
					latestTuningJob={latestTuningJob}
					applyStatus={applyStatus}
					onApplyBestParams={handleApplyBestParams}
					onBrowsePresets={() => setPresetModalOpen(true)}
					applyButtonDisabled={applyButtonDisabled}
					browsePresetsDisabled={browsePresetsDisabled}
					latestTuningRunNumber={latestTuningRunNumber}
					latestTuningRelative={latestTuningRelative}
					primaryPresetRelative={primaryPresetRelative}
					primaryPresetScoreLabel={primaryPresetScoreLabel}
					applyStatusColor={applyStatusColor}
					renderParameterField={renderParameterField}
					hyperparametersParameter={hyperparametersParameter}
				/>
			</div>

			{cvEnabled && (
				<p className="canvas-modal__note canvas-modal__note--muted">
					Cross-validation runs {cvFolds} folds before refitting on{' '}
					{cvRefitStrategy === 'train_plus_validation' ? 'training plus validation data.' : 'training data only.'}
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
						<div>
							<strong>⚠️ {scalingWarning.headline}</strong>
							<p style={{ margin: '0.35rem 0 0', fontSize: '0.85rem' }}>{scalingWarning.summary}</p>
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
								<li key={`scaling-tip-${index}`} style={{ marginBottom: '0.35rem' }}>
									{tip}
								</li>
							))}
						</ul>
					)}
				</div>
			)}

			<div className="canvas-modal__actions">
				<button
					type="button"
					className="btn btn-primary"
					onClick={handleEnqueueTraining}
					disabled={isActionDisabled}
				>
					{isCreatingJob ? 'Queuing…' : 'Launch training job'}
				</button>
			</div>

			<BestHyperparamsModal
				presets={hyperparamPresets}
				isOpen={isPresetModalOpen}
				onClose={() => setPresetModalOpen(false)}
				onApply={(preset) => {
					applyPreset(preset);
					setPresetModalOpen(false);
				}}
				activePresetId={lastAppliedPresetId}
				fieldMetadata={hyperparamFieldMap}
			/>
		</section>

		<TrainingJobHistory
			jobs={filteredJobs}
			isLoading={isJobsLoading}
			hiddenJobCount={hiddenJobCount}
			onRefresh={handleRefreshJobs}
			pipelineId={pipelineId}
			createJobError={createJobError instanceof Error ? createJobError : null}
			jobsError={jobsError}
			tuningJobsError={tuningJobsError}
			lastCreatedJob={lastCreatedJob}
			lastCreatedJobCount={lastCreatedJobCount}
			isTuningJobsLoading={isTuningJobsLoading}
		/>
		</>
	);
};
