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
import type { TrainModelCVConfig } from '../../hooks/useModelingConfiguration';

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
	onSaveDraftConfig?: () => void;
};

type PrimaryMetric = {
	label: string;
	value: number;
};

const STATUS_LABEL: Record<string, string> = {
	queued: 'Queued',
	running: 'Running',
	succeeded: 'Succeeded',
	failed: 'Failed',
	cancelled: 'Cancelled',
};

const METRIC_PREFERENCE = {
	classification: ['accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted'],
	regression: ['rmse', 'mae', 'r2'],
	fallback: ['rmse', 'mae', 'r2', 'accuracy', 'f1_weighted', 'roc_auc'],
};

const filterHyperparametersByFields = (
	values: Record<string, any> | null | undefined,
	fieldNames: Set<string>
): Record<string, any> => {
	if (!values) {
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

const sanitizeHyperparametersForPayload = (
	values: Record<string, any> | null | undefined,
	fieldNames: Set<string>
): Record<string, any> => {
	if (!values) {
		return {};
	}
	const sanitized: Record<string, any> = {};
	Object.entries(values).forEach(([key, value]) => {
		if (!fieldNames.has(key)) {
			return;
		}
		if (value === '' || value === null || value === undefined) {
			return;
		}
		sanitized[key] = value;
	});
	return sanitized;
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

const resolveMetricPreference = (metrics: Record<string, any> | null | undefined): string[] => {
	if (!metrics) {
		return METRIC_PREFERENCE.fallback;
	}
	const modelType = String(metrics.model_type ?? metrics.modelType ?? '').toLowerCase();
	if (modelType.includes('regressor')) {
		return METRIC_PREFERENCE.regression;
	}
	if (modelType.includes('classifier') || modelType.includes('classification')) {
		return METRIC_PREFERENCE.classification;
	}
	const hinted = String(metrics.problem_type ?? metrics.problemType ?? '').toLowerCase();
	if (hinted === 'regression') {
		return METRIC_PREFERENCE.regression;
	}
	if (hinted === 'classification') {
		return METRIC_PREFERENCE.classification;
	}
	return METRIC_PREFERENCE.fallback;
};

const pickPrimaryMetric = (metrics: Record<string, any> | null | undefined): PrimaryMetric | null => {
	if (!metrics) {
		return null;
	}

	const metricBuckets: Array<{ dataset: Record<string, any>; labelPrefix: string }> = [];
	const cvMean = metrics?.cross_validation?.metrics?.mean;
	if (isRecord(cvMean) && metrics?.cross_validation?.status === 'completed') {
		metricBuckets.push({ dataset: cvMean, labelPrefix: 'CV mean ' });
	}

	const candidateBuckets = [metrics.test, metrics.validation, metrics.train];
	for (const bucket of candidateBuckets) {
		if (isRecord(bucket)) {
			metricBuckets.push({ dataset: bucket, labelPrefix: '' });
		}
	}

	const orderedKeys = resolveMetricPreference(metrics);

	for (const bucket of metricBuckets) {
		for (const key of orderedKeys) {
			const value = bucket.dataset[key];
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	for (const bucket of metricBuckets) {
		for (const [key, value] of Object.entries(bucket.dataset)) {
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	return null;
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
	const [hyperparamValues, setHyperparamValues] = useState<Record<string, any>>({});
	const [showAdvanced, setShowAdvanced] = useState(false);
	const advancedInitializedRef = useRef(false);
	const [pipelineIdFromSavedConfig, setPipelineIdFromSavedConfig] = useState<string | null>(null);
	const [hasDraftChanges, setHasDraftChanges] = useState(false);
	const [lastAppliedTuningJobId, setLastAppliedTuningJobId] = useState<string | null>(null);
	const [applyStatus, setApplyStatus] = useState<{ message: string; tone: 'info' | 'warning' | 'success' } | null>(null);

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
		const sanitized = sanitizeHyperparametersForPayload(hyperparamValues, allowedHyperparamNames);
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
		onSuccess: (job) => {
			setLastCreatedJob(job);
			queryClient.invalidateQueries({ queryKey: trainingJobsQueryKey });
		},
	});

	const tuningJobsError = tuningJobsQuery.error as Error | null;
	const isTuningJobsLoading = tuningJobsQuery.isLoading || tuningJobsQuery.isFetching;
	const tuningJobs: HyperparameterTuningJobSummary[] = tuningJobsQuery.data?.jobs ?? [];
	
	// Get best parameters from the new dedicated API endpoint
	const bestParamsData = bestParamsQuery.data;
	const hasBestParams = Boolean(bestParamsData?.available && bestParamsData?.best_params);
	
	const bestParamsFromAPI = useMemo<Record<string, any> | null>(() => {
		if (!hasBestParams || !bestParamsData?.best_params) {
			return null;
		}
		if (allowedHyperparamNames.size === 0) {
			return null;
		}
		const filtered = filterHyperparametersByFields(bestParamsData.best_params, allowedHyperparamNames);
		return Object.keys(filtered).length ? filtered : null;
	}, [allowedHyperparamNames, bestParamsData, hasBestParams]);
	
	const latestTuningJob = useMemo<HyperparameterTuningJobSummary | null>(() => {
		if (!tuningJobs.length) {
			return null;
		}
		const succeeded = tuningJobs.filter((job) => job.status === 'succeeded');
		if (!succeeded.length) {
			return null;
		}
		return succeeded
			.slice()
			.sort((a, b) => {
				const aTime = Date.parse(a.updated_at || a.created_at || '0');
				const bTime = Date.parse(b.updated_at || b.created_at || '0');
				return bTime - aTime;
			})[0];
	}, [tuningJobs]);

	const bestParamsFromLatestTuning = useMemo<Record<string, any> | null>(() => {
		if (!latestTuningJob || !latestTuningJob.best_params) {
			return null;
		}
		if (allowedHyperparamNames.size === 0) {
			return null;
		}
		const filtered = filterHyperparametersByFields(latestTuningJob.best_params, allowedHyperparamNames);
		return Object.keys(filtered).length ? filtered : null;
	}, [allowedHyperparamNames, latestTuningJob]);

	// Use best params from API if available, otherwise fall back to latest tuning job
	const bestParamsToUse = bestParamsFromAPI || bestParamsFromLatestTuning;
	
	// Check if the best params are for the current model type
	const tuningModelMatches = Boolean(
		modelType && (
			(bestParamsData?.available && bestParamsData?.model_type === modelType) ||
			(latestTuningJob && latestTuningJob.model_type === modelType)
		)
	);

	const canApplyBestParams = Boolean(bestParamsToUse && tuningModelMatches);

	const applyButtonDisabled = !canApplyBestParams;

	// Use best params data if available, otherwise fall back to latest tuning job for display
	const displayJobInfo = bestParamsData?.available ? bestParamsData : latestTuningJob;
	const latestTuningTimestamp = displayJobInfo 
		? ('finished_at' in displayJobInfo 
			? displayJobInfo.finished_at 
			: ('updated_at' in displayJobInfo 
				? (displayJobInfo.updated_at || displayJobInfo.created_at) 
				: null))
		: null;
	const latestTuningRelative = latestTuningTimestamp ? formatRelativeTime(latestTuningTimestamp) : null;
	const latestTuningRunNumber = displayJobInfo
		? ('run_number' in displayJobInfo ? displayJobInfo.run_number : null)
		: null;
	const latestTuningJobId = displayJobInfo
		? ('job_id' in displayJobInfo 
			? displayJobInfo.job_id 
			: ('id' in displayJobInfo ? displayJobInfo.id : null))
		: null;

	useEffect(() => {
		if (!latestTuningJobId) {
			setLastAppliedTuningJobId(null);
			setApplyStatus(null);
			return;
		}
		if (lastAppliedTuningJobId && latestTuningJobId !== lastAppliedTuningJobId) {
			setApplyStatus(null);
		}
	}, [lastAppliedTuningJobId, latestTuningJobId]);

	const handleApplyBestParams = useCallback(() => {
		if (!latestTuningJob) {
			setApplyStatus({
				message: 'No tuning job results available to apply.',
				tone: 'warning',
			});
			return;
		}
		if (!bestParamsFromLatestTuning) {
			setApplyStatus({
				message:
					'Latest tuning run did not produce compatible hyperparameters for this model template.',
				tone: 'warning',
			});
			return;
		}
		if (!tuningModelMatches) {
			setApplyStatus({
				message: `Latest tuning run targeted “${latestTuningJob.model_type}”. Switch the model template to apply its parameters.`,
				tone: 'warning',
			});
			return;
		}

		setHyperparamValues(() => cloneJson(bestParamsToUse));
		setShowAdvanced(true);
		advancedInitializedRef.current = true;
		setLastAppliedTuningJobId(latestTuningJobId || '');
		setApplyStatus({
			message: `Applied best parameters from tuning run${latestTuningRunNumber ? ` ${latestTuningRunNumber}` : ''}.`,
			tone: 'success',
		});
	}, [
		bestParamsToUse,
		bestParamsData,
		latestTuningJob,
		latestTuningJobId,
		latestTuningRunNumber,
		tuningModelMatches,
	]);

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
								<option key={opt.value} value={opt.value}>
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
			onSaveDraftConfig();
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

		const payload = {
			dataset_source_id: sourceId,
			pipeline_id: pipelineId,
			node_id: nodeId,
			model_type: modelType,
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

		const renderJobSummary = (job: TrainingJobSummary) => {
			const metric = pickPrimaryMetric(job.metrics ?? null);
			const updatedLabel = job.updated_at ? formatRelativeTime(job.updated_at) : null;
			const fallbackUpdated = job.updated_at || job.created_at;
			const timestampLabel = fallbackUpdated
				? updatedLabel ?? new Date(fallbackUpdated).toLocaleString()
				: null;
			const statusLabel = STATUS_LABEL[job.status] ?? job.status;
			const jobProblemType = (
				job.problem_type || job.metadata?.resolved_problem_type || job.metadata?.problem_type || ''
			).toString();
			const cvMetadata = (job.metadata?.cross_validation ?? null) as Record<string, any> | null;
			const cvMetrics = (job.metrics?.cross_validation ?? null) as Record<string, any> | null;

			const detailParts: string[] = [];
			detailParts.push(`Model ${job.model_type}`);
			if (jobProblemType) {
				detailParts.push(jobProblemType.charAt(0).toUpperCase() + jobProblemType.slice(1));
			}
			if (cvMetadata?.enabled) {
				const strategyKey = String(cvMetadata.strategy ?? 'auto');
				const strategyLabel =
					strategyKey === 'stratified_kfold'
						? 'Stratified KFold'
						: strategyKey === 'kfold'
						? 'KFold'
						: 'Auto CV';
				const foldLabel = typeof cvMetadata.folds === 'number' ? `${cvMetadata.folds} folds` : 'CV enabled';
				detailParts.push(`${strategyLabel} (${foldLabel})`);
				if (cvMetrics?.status === 'skipped') {
					const reason = String(cvMetrics?.reason ?? 'unknown reason').replace(/_/g, ' ');
					detailParts.push(`CV skipped (${reason})`);
				}
			} else if (cvMetrics?.status === 'skipped') {
				const reason = String(cvMetrics?.reason ?? 'unknown reason').replace(/_/g, ' ');
				detailParts.push(`CV skipped (${reason})`);
			}
			if (metric) {
				detailParts.push(`${metric.label}: ${formatMetricValue(metric.value)}`);
			}
			if (timestampLabel) {
				detailParts.push(`Updated ${timestampLabel}`);
			}

			return (
				<li key={job.id}>
					<strong>v{job.version}</strong> — {statusLabel}
					{detailParts.length > 0 ? ` • ${detailParts.join(' • ')}` : ''}
				</li>
			);
		};

	return (
		<section className="canvas-modal__section">
			<div className="canvas-modal__section-header">
				<h3>Background training jobs</h3>
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
				Enqueue a background training job for this node. Jobs run via Celery, versioned per pipeline, and
				report metrics when finished.
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
					Training job {lastCreatedJob.id} queued (version {lastCreatedJob.version}).
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
				
				{/* Hyperparameter configuration mode toggle */}
				{modelType && hyperparamFields.length > 0 && (
					<div className="canvas-modal__parameter" style={{ gridColumn: '1 / -1' }}>
						<div
							style={{
								display: 'flex',
								flexWrap: 'wrap',
								alignItems: 'flex-start',
								justifyContent: 'space-between',
								gap: '1rem',
								marginBottom: '0.75rem',
								padding: '0.75rem 1rem',
								background: 'rgba(15, 23, 42, 0.35)',
								borderRadius: '6px',
								border: '1px solid rgba(148, 163, 184, 0.25)'
							}}
						>
							<div style={{ flex: '1 1 260px' }}>
								<strong style={{ fontSize: '0.95rem', color: '#e2e8f0' }}>Model Configuration</strong>
								<p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: 'rgba(148, 163, 184, 0.85)' }}>
									{showAdvanced
										? 'Fine-tune hyperparameters or use defaults for quick training'
										: 'Using default hyperparameters — toggle Advanced to customize'}
								</p>
								{displayJobInfo && (
									<p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: 'rgba(148, 163, 184, 0.8)' }}>
										{hasBestParams ? (
											<>
												✓ Tuned parameters available for <strong>{modelType}</strong>
												{latestTuningRunNumber && ` (run ${latestTuningRunNumber})`}
												{latestTuningRelative && ` • ${latestTuningRelative}`}
											</>
										) : (
											<>
												Latest tuning run {latestTuningRunNumber || 'N/A'}
												{latestTuningRelative ? ` • ${latestTuningRelative}` : ''}
												{tuningModelMatches
													? ' matches the current model template.'
													: displayJobInfo && 'model_type' in displayJobInfo 
													? ` targeted ${displayJobInfo.model_type}.`
													: latestTuningJob
													? ` targeted ${latestTuningJob.model_type}.`
													: ''}
											</>
										)}
									</p>
								)}
								{applyStatus && (
									<p style={{ margin: '0.5rem 0 0 0', fontSize: '0.8rem', color: applyStatusColor }}>
										{applyStatus.message}
									</p>
								)}
							</div>
							<div
								style={{
									display: 'flex',
									flexWrap: 'wrap',
									gap: '0.5rem',
									alignItems: 'center',
									justifyContent: 'flex-end'
								}}
							>
								{displayJobInfo && (
									<button
										type="button"
										onClick={handleApplyBestParams}
										disabled={applyButtonDisabled}
										style={{
											padding: '0.5rem 1rem',
											fontSize: '0.875rem',
											fontWeight: 500,
											color: applyButtonDisabled ? 'rgba(148, 163, 184, 0.55)' : '#0c4a6e',
											background: applyButtonDisabled ? 'rgba(15, 23, 42, 0.35)' : 'rgba(191, 219, 254, 0.9)',
											border: applyButtonDisabled ? '1px solid rgba(148, 163, 184, 0.35)' : '1px solid rgba(147, 197, 253, 0.9)',
											borderRadius: '4px',
											cursor: applyButtonDisabled ? 'not-allowed' : 'pointer',
											transition: 'all 0.2s ease',
											minWidth: '130px',
											boxShadow: applyButtonDisabled ? 'none' : '0 3px 10px rgba(14, 116, 144, 0.25)'
										}}
									>
										Apply best params
									</button>
								)}
								<button
									type="button"
									onClick={handleToggleAdvanced}
									style={{
										padding: '0.5rem 1rem',
										fontSize: '0.875rem',
										fontWeight: '500',
										color: showAdvanced ? '#fff' : 'rgba(148, 163, 184, 0.85)',
										background: showAdvanced ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(15, 23, 42, 0.6)',
										border: showAdvanced ? 'none' : '1px solid rgba(148, 163, 184, 0.45)',
										borderRadius: '4px',
										cursor: 'pointer',
										transition: 'all 0.2s ease',
										minWidth: '100px',
										boxShadow: showAdvanced ? '0 4px 12px rgba(118, 75, 162, 0.25)' : 'none'
									}}
									onMouseEnter={(e) => {
										if (showAdvanced) {
											e.currentTarget.style.boxShadow = '0 6px 16px rgba(118, 75, 162, 0.35)';
										} else {
											e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.65)';
											e.currentTarget.style.background = 'rgba(15, 23, 42, 0.8)';
										}
									}}
									onMouseLeave={(e) => {
										if (showAdvanced) {
											e.currentTarget.style.boxShadow = '0 4px 12px rgba(118, 75, 162, 0.25)';
										} else {
											e.currentTarget.style.borderColor = 'rgba(148, 163, 184, 0.45)';
											e.currentTarget.style.background = 'rgba(15, 23, 42, 0.6)';
										}
									}}
								>
									{showAdvanced ? '✓ Advanced' : 'Default'}
								</button>
							</div>
						</div>
					</div>
				)}
				
				{/* Render dynamic hyperparameter fields */}
				{modelType && hyperparamQuery.isLoading && (
					<p className="canvas-modal__note canvas-modal__note--muted">Loading hyperparameters…</p>
				)}
				{modelType && hyperparamQuery.error && (
					<p className="canvas-modal__note canvas-modal__note--error">
						Failed to load hyperparameters: {hyperparamQuery.error.message}
					</p>
				)}
				{modelType && showAdvanced && hyperparamFields.length > 0 && (
					<div style={{ 
						gridColumn: '1 / -1',
						padding: '1.25rem',
						background: 'rgba(15, 23, 42, 0.25)',
						border: '1px solid rgba(148, 163, 184, 0.2)',
						borderRadius: '6px',
						marginTop: '0.5rem'
					}}>
						<div style={{ 
							marginBottom: '1rem',
							paddingBottom: '0.75rem',
							borderBottom: '2px solid rgba(102, 126, 234, 0.3)'
						}}>
							<h4 style={{ 
								margin: '0 0 0.5rem 0', 
								fontSize: '1rem',
								fontWeight: '600',
								color: '#e2e8f0'
							}}>
								Advanced Hyperparameters
							</h4>
							<p style={{ 
								margin: '0', 
								fontSize: '0.875rem', 
								color: 'rgba(148, 163, 184, 0.85)',
								lineHeight: '1.5'
							}}>
								Configure {modelType.replace(/_/g, ' ')} hyperparameters. Leave empty to use recommended defaults.
							</p>
						</div>
						<div style={{ 
							display: 'grid',
							gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
							gap: '1rem'
						}}>
							{hyperparamFields.map(renderHyperparamField)}
						</div>
					</div>
				)}
				{!modelType && hyperparametersParameter && renderParameterField(hyperparametersParameter)}
			</div>

			{cvEnabled && (
				<p className="canvas-modal__note canvas-modal__note--muted">
					Cross-validation runs {cvFolds} folds before refitting on{' '}
					{cvRefitStrategy === 'train_plus_validation' ? 'training plus validation data.' : 'training data only.'}
				</p>
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

					{isJobsLoading && <p className="canvas-modal__note canvas-modal__note--muted">Loading recent jobs…</p>}
					{hiddenJobCount > 0 && (
						<p className="canvas-modal__note canvas-modal__note--muted">
							{hiddenJobCount === 1
								? '1 training job uses a different problem type. Switch the problem type to view it.'
								: `${hiddenJobCount} training jobs use a different problem type. Switch the problem type to view them.`}
						</p>
					)}
					{!isJobsLoading && filteredJobs.length === 0 && (
						<p className="canvas-modal__note canvas-modal__note--muted">No training jobs found for this node.</p>
					)}
					{filteredJobs.length > 0 && <ul className="canvas-modal__note-list">{filteredJobs.map(renderJobSummary)}</ul>}
		</section>
	);
};
