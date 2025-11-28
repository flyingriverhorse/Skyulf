import React, { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import type {
	FeatureGraph,
	FeatureNodeParameter,
	TrainingJobListResponse,
	TrainingJobSummary,
	FetchTrainingJobsOptions,
} from '../../../../api';
import { fetchTrainingJobs, generatePipelineId } from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';

type ModelRegistrySectionProps = {
	nodeId: string;
	sourceId?: string | null;
	graph: FeatureGraph | null;
	parameters: FeatureNodeParameter[];
	config: Record<string, any> | null;
	renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
};

type PrimaryMetric = {
	label: string;
	value: number;
};

type MetricBucket = {
	label: string | null;
	values: Record<string, any>;
};

type NormalizedJob = {
	job: TrainingJobSummary;
	problemType: 'classification' | 'regression' | 'unknown';
	modelLabel: string;
	primaryMetric: PrimaryMetric | null;
	metricBucket: MetricBucket;
};

type MethodSummary = {
	modelType: string;
	label: string;
	bestJob: NormalizedJob | null;
};

const STATUS_LABEL: Record<TrainingJobSummary['status'], string> = {
	queued: 'Queued',
	running: 'Running',
	succeeded: 'Succeeded',
	failed: 'Failed',
	cancelled: 'Cancelled',
};

const STATUS_TONE: Record<TrainingJobSummary['status'], 'neutral' | 'info' | 'success' | 'danger'> = {
	queued: 'info',
	running: 'info',
	succeeded: 'success',
	failed: 'danger',
	cancelled: 'neutral',
};

const METRIC_PREFERENCE = {
	classification: [
		'accuracy',
		'f1_weighted',
		'roc_auc',
		'roc_auc_weighted',
		'precision_weighted',
		'recall_weighted',
		'pr_auc',
		'pr_auc_weighted',
		'g_score',
		'f1',
		'precision',
		'recall',
	],
	regression: ['rmse', 'mae', 'r2', 'mape', 'mse'],
	fallback: ['rmse', 'mae', 'r2', 'accuracy', 'f1_weighted', 'roc_auc', 'f1', 'precision', 'recall'],
};

const CLASSIFICATION_METRICS = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc', 'roc_auc_weighted', 'f1', 'precision', 'recall'];
const REGRESSION_METRICS = ['rmse', 'mae', 'r2', 'mape', 'mse'];

const LOWER_IS_BETTER = new Set(['rmse', 'mae', 'mape', 'mse']);

const MODEL_LABEL_OVERRIDES: Record<string, string> = {
	logistic_regression: 'Logistic Regression',
	random_forest_classifier: 'Random Forest (Classifier)',
	random_forest_regressor: 'Random Forest (Regressor)',
	ridge_regression: 'Ridge Regression',
};

const isRecord = (value: unknown): value is Record<string, any> =>
	Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const parseConfigBoolean = (value: unknown, fallback: boolean): boolean => {
	if (value === null || value === undefined) {
		return fallback;
	}
	if (typeof value === 'boolean') {
		return value;
	}
	if (typeof value === 'string') {
		const normalized = value.trim().toLowerCase();
		if (!normalized) {
			return fallback;
		}
		if (['true', '1', 'yes', 'on', 'enabled'].includes(normalized)) {
			return true;
		}
		if (['false', '0', 'no', 'off', 'disabled'].includes(normalized)) {
			return false;
		}
	}
	if (typeof value === 'number') {
		if (!Number.isFinite(value)) {
			return fallback;
		}
		return value !== 0;
	}
	return Boolean(value);
};

const formatMetricValue = (value: number): string => {
	if (!Number.isFinite(value)) {
		return '—';
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

const resolveMetricPreference = (metrics: Record<string, any> | null | undefined): string[] => {
	if (!metrics) {
		return METRIC_PREFERENCE.fallback;
	}

	const modelType = String(metrics.model_type ?? metrics.modelType ?? '').toLowerCase();
	if (modelType.includes('regress')) {
		return METRIC_PREFERENCE.regression;
	}
	if (modelType.includes('class') || modelType.includes('classification')) {
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
	if (!isRecord(metrics)) {
		return null;
	}

	const metricBuckets: Array<{ values: Record<string, any>; labelPrefix: string }> = [];
	const cvMean = metrics?.cross_validation?.metrics?.mean;
	if (isRecord(cvMean) && metrics?.cross_validation?.status === 'completed') {
		metricBuckets.push({ values: cvMean, labelPrefix: 'CV mean ' });
	}
	if (isRecord(metrics.test)) {
		metricBuckets.push({ values: metrics.test, labelPrefix: '' });
	}
	if (isRecord(metrics.validation)) {
		metricBuckets.push({ values: metrics.validation, labelPrefix: '' });
	}
	if (isRecord(metrics.train)) {
		metricBuckets.push({ values: metrics.train, labelPrefix: '' });
	}

	const orderedKeys = resolveMetricPreference(metrics);

	for (const bucket of metricBuckets) {
		for (const key of orderedKeys) {
			const value = bucket.values[key];
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	for (const bucket of metricBuckets) {
		for (const [key, value] of Object.entries(bucket.values)) {
			if (typeof value === 'number' && Number.isFinite(value)) {
				const label = bucket.labelPrefix ? `${bucket.labelPrefix}${key}` : key;
				return { label, value };
			}
		}
	}

	return null;
};

const selectMetricBucket = (metrics: Record<string, any> | null | undefined): MetricBucket => {
	if (!isRecord(metrics)) {
		return { label: null, values: {} };
	}
	if (isRecord(metrics.test)) {
		return { label: 'Test', values: metrics.test };
	}
	if (isRecord(metrics.validation)) {
		return { label: 'Validation', values: metrics.validation };
	}
	const cvMean = metrics?.cross_validation?.metrics?.mean;
	if (isRecord(cvMean)) {
		return { label: 'CV mean', values: cvMean };
	}
	if (isRecord(metrics.train)) {
		return { label: 'Train', values: metrics.train };
	}
	return { label: null, values: {} };
};

const formatModelType = (modelType: string): string => {
	if (!modelType) {
		return 'Unknown model';
	}
	const normalized = modelType.trim();
	if (MODEL_LABEL_OVERRIDES[normalized]) {
		return MODEL_LABEL_OVERRIDES[normalized];
	}
	return normalized
		.split('_')
		.filter(Boolean)
		.map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
		.join(' ');
};

const normalizeProblemType = (job: TrainingJobSummary): 'classification' | 'regression' | 'unknown' => {
	const metadata = (job.metadata ?? {}) as Record<string, any>;
	const candidates: Array<unknown> = [job.problem_type, metadata?.resolved_problem_type, metadata?.problem_type];

	for (const candidate of candidates) {
		if (typeof candidate === 'string') {
			const normalized = candidate.trim().toLowerCase();
			if (normalized === 'classification' || normalized === 'regression') {
				return normalized;
			}
		}
	}

	const modelType = typeof job.model_type === 'string' ? job.model_type.toLowerCase() : '';
	if (modelType.includes('classifier') || modelType.includes('classification')) {
		return 'classification';
	}
	if (modelType.includes('regress') || modelType.includes('regression')) {
		return 'regression';
	}

	const metrics = job.metrics as Record<string, any> | null;
	if (isRecord(metrics)) {
		const metricKeys = new Set<string>();
		const collectKeys = (bucket: Record<string, any> | null | undefined) => {
			if (!isRecord(bucket)) {
				return;
			}
			Object.keys(bucket).forEach((key) => metricKeys.add(key.toLowerCase()));
		};

		collectKeys(metrics.test);
		collectKeys(metrics.validation);
		collectKeys(metrics.train);
		const cvMean = metrics?.cross_validation?.metrics?.mean;
		if (isRecord(cvMean)) {
			collectKeys(cvMean);
		}

		const hasClassification = CLASSIFICATION_METRICS.some((metric) => metricKeys.has(metric));
		const hasRegression = REGRESSION_METRICS.some((metric) => metricKeys.has(metric));

		if (hasClassification && !hasRegression) {
			return 'classification';
		}
		if (hasRegression && !hasClassification) {
			return 'regression';
		}
	}

	return 'unknown';
};

const normalizeJobs = (jobs: TrainingJobSummary[]): NormalizedJob[] =>
	jobs.map((job) => {
		const metrics = (job.metrics ?? null) as Record<string, any> | null;
		return {
			job,
			problemType: normalizeProblemType(job),
			modelLabel: formatModelType(job.model_type),
			primaryMetric: pickPrimaryMetric(metrics),
			metricBucket: selectMetricBucket(metrics),
		};
	});

const coefficientForMetric = (metric: PrimaryMetric | null): number | null => {
	if (!metric || !Number.isFinite(metric.value)) {
		return null;
	}
	const label = metric.label.toLowerCase();
	const lowerBetter = LOWER_IS_BETTER.has(label) || LOWER_IS_BETTER.has(label.replace(/^cv mean\s+/i, ''));
	return lowerBetter ? -metric.value : metric.value;
};

const ModelRegistrySection: React.FC<ModelRegistrySectionProps> = ({
	nodeId,
	sourceId,
	graph,
	parameters,
	config,
	renderParameterField,
}) => {
	const graphPayload = useMemo<FeatureGraph | null>(() => {
		if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
			return null;
		}
		return {
			nodes: graph.nodes,
			edges: graph.edges,
		};
	}, [graph]);

	const upstreamTrainingNodeIds = useMemo(() => {
		if (!graphPayload) {
			return [] as string[];
		}

		const nodeLookup = new Map<string, any>();
		graphPayload.nodes.forEach((node: any) => {
			if (node && node.id) {
				nodeLookup.set(node.id, node);
			}
		});

		const directTrainNodes = new Set<string>();
		graphPayload.edges.forEach((edge: any) => {
			if (!edge || edge.target !== nodeId) {
				return;
			}
			const sourceNode = nodeLookup.get(edge.source);
			if (sourceNode?.data?.catalogType === 'train_model_draft') {
				directTrainNodes.add(sourceNode.id);
			}
		});

		if (directTrainNodes.size > 0) {
			return Array.from(directTrainNodes);
		}

		const anyTrainNodes = graphPayload.nodes
			.filter((node: any) => node?.data?.catalogType === 'train_model_draft')
			.map((node: any) => node.id);

		return anyTrainNodes;
	}, [graphPayload, nodeId]);

	const upstreamKey = useMemo(() => {
		if (!upstreamTrainingNodeIds.length) {
			return 'all';
		}
		return upstreamTrainingNodeIds.slice().sort().join('|');
	}, [upstreamTrainingNodeIds]);

	const [pipelineId, setPipelineId] = useState<string | null>(null);
	const [pipelineError, setPipelineError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		setPipelineError(null);

		if (!sourceId || !graphPayload) {
			setPipelineId(null);
			return () => {
				cancelled = true;
			};
		}

		generatePipelineId(sourceId, graphPayload)
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
	}, [graphPayload, sourceId]);

	const showNonSuccess = parseConfigBoolean(config?.show_non_success, true);

	const trainingJobsQueryKey = useMemo(
		() => ['feature-canvas', 'model-registry', sourceId ?? 'none', pipelineId ?? 'pending', upstreamKey, showNonSuccess ? 'all-statuses' : 'succeeded'],
		[pipelineId, showNonSuccess, sourceId, upstreamKey]
	);

	const trainingJobsQuery = useQuery<TrainingJobListResponse, Error>({
		queryKey: trainingJobsQueryKey,
		queryFn: async () => {
			const baseParams: FetchTrainingJobsOptions = {
				datasetSourceId: sourceId || undefined,
				limit: 50,
			};

			const aggregatedJobs = new Map<string, TrainingJobSummary>();
			const encounteredErrors: Error[] = [];
			let datasetResponse: TrainingJobListResponse | null = null;

			const appendJobs = (jobs?: TrainingJobSummary[] | null) => {
				if (!Array.isArray(jobs) || !jobs.length) {
					return;
				}
				jobs.forEach((job) => {
					if (job && !aggregatedJobs.has(job.id)) {
						aggregatedJobs.set(job.id, job);
					}
				});
			};

			const safeFetch = async (params: FetchTrainingJobsOptions) => {
				try {
					const response = await fetchTrainingJobs(params);
					appendJobs(response?.jobs);
					return response;
				} catch (error) {
					if (error instanceof Error) {
						encounteredErrors.push(error);
					}
					return null;
				}
			};

			if (pipelineId) {
				await safeFetch({ ...baseParams, pipelineId });
			}

			if (upstreamTrainingNodeIds.length === 1) {
				const [targetNode] = upstreamTrainingNodeIds;
				await safeFetch({ ...baseParams, nodeId: targetNode });
			}

			datasetResponse = await safeFetch(baseParams);

			if (!aggregatedJobs.size) {
				if (datasetResponse) {
					return datasetResponse;
				}
				if (encounteredErrors.length) {
					throw encounteredErrors[0];
				}
				return { jobs: [], total: 0 };
			}

			const jobs = Array.from(aggregatedJobs.values());
			const total = datasetResponse?.total ?? jobs.length;
			return { jobs, total };
		},
		enabled: Boolean(sourceId),
		staleTime: 30 * 1000,
		retry: (failureCount, error) => {
			if (error.message.includes('Sign in')) {
				return false;
			}
			return failureCount < 2;
		},
		refetchInterval: (query) => {
			const currentData = query.state.data as TrainingJobListResponse | undefined;
			const jobList = currentData?.jobs ?? [];
			const hasActiveJob = jobList.some((job) => job.status === 'queued' || job.status === 'running');
			return hasActiveJob ? 5000 : false;
		},
	});

	const allJobs = trainingJobsQuery.data?.jobs ?? [];

	const scopedJobs = useMemo(() => {
		if (!allJobs.length) {
			return [] as TrainingJobSummary[];
		}

		if (!upstreamTrainingNodeIds.length) {
			return allJobs;
		}

		const filtered = allJobs.filter((job) => upstreamTrainingNodeIds.includes(job.node_id));
		return filtered.length ? filtered : allJobs;
	}, [allJobs, upstreamTrainingNodeIds]);

	const normalizedJobs = useMemo(() => normalizeJobs(scopedJobs), [scopedJobs]);

	const statusScopedJobs = useMemo(() => {
		if (showNonSuccess) {
			return normalizedJobs;
		}
		return normalizedJobs.filter((entry) => entry.job.status === 'succeeded');
	}, [normalizedJobs, showNonSuccess]);

	const classificationJobs = useMemo(
		() => statusScopedJobs.filter((entry) => entry.problemType === 'classification'),
		[statusScopedJobs]
	);
	const regressionJobs = useMemo(
		() => statusScopedJobs.filter((entry) => entry.problemType === 'regression'),
		[statusScopedJobs]
	);

	const fallbackProblemType = useMemo<'classification' | 'regression'>(() => {
		if (!classificationJobs.length && !regressionJobs.length) {
			return 'classification';
		}
		if (!classificationJobs.length) {
			return 'regression';
		}
		if (!regressionJobs.length) {
			return 'classification';
		}
		return classificationJobs.length >= regressionJobs.length ? 'classification' : 'regression';
	}, [classificationJobs.length, regressionJobs.length]);

	const defaultProblemSetting = typeof config?.default_problem_type === 'string'
		? config.default_problem_type.trim().toLowerCase()
		: 'auto';

	const resolvedDefaultProblemType: 'classification' | 'regression' =
		defaultProblemSetting === 'classification' || defaultProblemSetting === 'regression'
			? defaultProblemSetting
			: fallbackProblemType;

	const [activeProblemType, setActiveProblemType] = useState<'classification' | 'regression'>(resolvedDefaultProblemType);

	useEffect(() => {
		setActiveProblemType(resolvedDefaultProblemType);
	}, [resolvedDefaultProblemType]);

	const selectedJobs = activeProblemType === 'classification' ? classificationJobs : regressionJobs;

	const methodOptions = useMemo(
		() => {
			const counts = new Map<string, { label: string; count: number }>();
			selectedJobs.forEach((entry) => {
				const key = entry.job.model_type;
				const existing = counts.get(key) ?? { label: entry.modelLabel, count: 0 };
				existing.count += 1;
				counts.set(key, existing);
			});
			return Array.from(counts.entries())
				.map(([value, meta]) => ({ value, label: meta.label, count: meta.count }))
				.sort((a, b) => a.label.localeCompare(b.label));
		},
		[selectedJobs]
	);

	const defaultMethodSetting = typeof config?.default_method === 'string' ? config.default_method.trim().toLowerCase() : '';

	const [activeMethod, setActiveMethod] = useState<string>('all');

	useEffect(() => {
		if (!methodOptions.length) {
			setActiveMethod('all');
			return;
		}
		const matched = methodOptions.find((option) => option.value.toLowerCase() === defaultMethodSetting);
		setActiveMethod(matched ? matched.value : 'all');
	}, [defaultMethodSetting, methodOptions, activeProblemType]);

	const filteredJobs = useMemo(() => {
		if (activeMethod === 'all') {
			return selectedJobs;
		}
		return selectedJobs.filter((entry) => entry.job.model_type === activeMethod);
	}, [activeMethod, selectedJobs]);

	const sortedJobs = useMemo(() => {
		return filteredJobs
			.slice()
			.sort((a, b) => {
				if (b.job.version !== a.job.version) {
					return b.job.version - a.job.version;
				}
				const aTime = Date.parse(a.job.updated_at ?? a.job.created_at ?? '');
				const bTime = Date.parse(b.job.updated_at ?? b.job.created_at ?? '');
				return bTime - aTime;
			});
	}, [filteredJobs]);

	const methodSummaries = useMemo<MethodSummary[]>(() => {
		if (!selectedJobs.length) {
			return [];
		}

		const groups = new Map<string, NormalizedJob[]>();
		selectedJobs.forEach((entry) => {
			const key = entry.job.model_type;
			const bucket = groups.get(key) ?? [];
			bucket.push(entry);
			groups.set(key, bucket);
		});

		const summaries: MethodSummary[] = [];
		groups.forEach((jobsForMethod, modelType) => {
			const bestJob = jobsForMethod.reduce<NormalizedJob | null>((best, candidate) => {
				const candidateScore = coefficientForMetric(candidate.primaryMetric);
				if (candidateScore === null) {
					return best;
				}
				if (!best) {
					return candidate;
				}
				const bestScore = coefficientForMetric(best.primaryMetric);
				if (bestScore === null) {
					return candidate;
				}
				return candidateScore > bestScore ? candidate : best;
			}, null);

			summaries.push({
				modelType,
				label: formatModelType(modelType),
				bestJob,
			});
		});

		return summaries.sort((a, b) => a.label.localeCompare(b.label));
	}, [selectedJobs]);

	const registryParameters = useMemo(
		() =>
			parameters.filter((parameter) =>
				['default_problem_type', 'show_non_success'].includes(parameter.name)
			),
		[parameters]
	);

	const metricColumns = activeProblemType === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS;

	const isLoading = trainingJobsQuery.isLoading || trainingJobsQuery.isFetching;
	const loadError: Error | null = trainingJobsQuery.error ?? null;
	const hasJobs = sortedJobs.length > 0;

	return (
		<section className="canvas-modal__section">
			<div className="canvas-modal__section-header">
				<h3>Model registry</h3>
				{pipelineId && <span className="model-registry__pipeline">Pipeline ID: {pipelineId}</span>}
			</div>

			{pipelineError && (
				<p className="canvas-modal__note canvas-modal__note--error">{pipelineError}</p>
			)}

			{!upstreamTrainingNodeIds.length && (
				<p className="canvas-modal__note canvas-modal__note--warning">
					Connect this node to a “Train model” step to populate the registry.
				</p>
			)}

			{registryParameters.length > 0 && (
				<div className="canvas-modal__parameter-grid model-registry__parameters">
					{registryParameters.map((parameter) => (
						<React.Fragment key={parameter.name}>{renderParameterField(parameter)}</React.Fragment>
					))}
				</div>
			)}

			<div className="model-registry__toolbar">
				<div className="model-registry__tabs" role="tablist" aria-label="Problem type">
					{(['classification', 'regression'] as const).map((tab) => (
						<button
							key={tab}
							type="button"
							role="tab"
							data-active={activeProblemType === tab}
							className="model-registry__tab"
							onClick={() => setActiveProblemType(tab)}
							disabled={(tab === 'classification' && !classificationJobs.length) || (tab === 'regression' && !regressionJobs.length)}
						>
							{tab === 'classification' ? 'Classification' : 'Regression'}
							<span className="model-registry__tab-count">
								{tab === 'classification' ? classificationJobs.length : regressionJobs.length}
							</span>
						</button>
					))}
				</div>

				<div className="model-registry__methods" aria-label="Model template filter">
					<button
						type="button"
						className="model-registry__method"
						data-active={activeMethod === 'all'}
						onClick={() => setActiveMethod('all')}
					>
						All methods
						<span className="model-registry__method-count">{selectedJobs.length}</span>
					</button>
					{methodOptions.map((option) => (
						<button
							key={option.value}
							type="button"
							className="model-registry__method"
							data-active={activeMethod === option.value}
							onClick={() => setActiveMethod(option.value)}
						>
							{option.label}
							<span className="model-registry__method-count">{option.count}</span>
						</button>
					))}
				</div>
			</div>

			{methodSummaries.length > 0 && (
				<div className="model-registry__summary">
					{methodSummaries.map((summary) => {
						const best = summary.bestJob;
						const metricLabel = best?.primaryMetric ? best.primaryMetric.label : null;
						const metricValue = best?.primaryMetric ? formatMetricValue(best.primaryMetric.value) : '—';
						return (
							<div key={summary.modelType} className="model-registry__summary-card">
								<span className="model-registry__summary-title">{summary.label}</span>
								<span className="model-registry__summary-metric">
									{metricLabel || 'No metrics'}
									{metricLabel && <strong>{metricValue}</strong>}
								</span>
								{best?.job && (
									<span className="model-registry__summary-version">v{best.job.version}</span>
								)}
							</div>
						);
					})}
				</div>
			)}

			{isLoading && <p className="canvas-modal__note canvas-modal__note--muted">Loading model history…</p>}
			{loadError && <p className="canvas-modal__note canvas-modal__note--error">{loadError.message}</p>}

			{!isLoading && !loadError && !hasJobs && (
				<p className="canvas-modal__note canvas-modal__note--muted">
					No model versions available for the current filters.
				</p>
			)}

			{hasJobs && (
				<div className="model-registry__table-wrapper" style={{ overflowX: 'auto' }}>
					<table className="model-registry__table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
						<thead>
							<tr>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'left', color: '#e2e8f0', fontWeight: 600 }}>Version</th>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'left', color: '#e2e8f0', fontWeight: 600 }}>Status</th>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'left', color: '#e2e8f0', fontWeight: 600 }}>Model</th>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'left', color: '#e2e8f0', fontWeight: 600 }}>Target</th>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'right', color: '#e2e8f0', fontWeight: 600 }}>Primary metric</th>
								{metricColumns.map((metric) => (
									<th key={metric} scope="col" style={{ padding: '0.75rem', textAlign: 'right', color: '#e2e8f0', fontWeight: 600 }}>
										{metric.replace(/_/g, ' ')}
									</th>
								))}
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'left', color: '#e2e8f0', fontWeight: 600 }}>Evaluated</th>
								<th scope="col" style={{ padding: '0.75rem', textAlign: 'right', color: '#e2e8f0', fontWeight: 600 }}>Updated</th>
							</tr>
						</thead>
						<tbody>
											{sortedJobs.map((entry) => {
								const { job, primaryMetric, metricBucket } = entry;
								const bucketValues = metricBucket.values || {};
								const bucketLabel = metricBucket.label ?? '—';
								const updatedAt = job.updated_at ?? job.created_at ?? null;
								return (
													<tr key={job.id} style={{ borderTop: '1px solid rgba(148, 163, 184, 0.1)' }}>
										<td style={{ padding: '0.75rem', color: 'rgba(148, 163, 184, 0.8)' }}>v{job.version}</td>
										<td style={{ padding: '0.75rem' }}>
											<span
												style={{
													display: 'inline-block',
													padding: '0.15rem 0.5rem',
													borderRadius: '4px',
													fontSize: '0.75rem',
													fontWeight: 500,
													background:
														job.status === 'succeeded'
															? 'rgba(34, 197, 94, 0.15)'
															: job.status === 'failed'
															? 'rgba(239, 68, 68, 0.15)'
															: 'rgba(59, 130, 246, 0.15)',
													color:
														job.status === 'succeeded'
															? '#4ade80'
															: job.status === 'failed'
															? '#f87171'
															: '#60a5fa',
												}}
											>
												{STATUS_LABEL[job.status]}
											</span>
										</td>
										<td style={{ padding: '0.75rem', fontWeight: 500, color: '#e2e8f0' }}>{entry.modelLabel}</td>
										<td style={{ padding: '0.75rem', color: 'rgba(148, 163, 184, 0.8)' }}>{job.metadata?.target_column || '—'}</td>
										<td style={{ padding: '0.75rem', textAlign: 'right' }}>
											{primaryMetric ? (
												<span className="model-registry__metric">
													<span className="model-registry__metric-label" style={{ marginRight: '0.5rem', color: 'rgba(148, 163, 184, 0.6)', fontSize: '0.75rem' }}>{primaryMetric.label}</span>
													<span className="model-registry__metric-value" style={{ fontFamily: 'monospace', color: '#e2e8f0', fontWeight: 600 }}>{formatMetricValue(primaryMetric.value)}</span>
												</span>
											) : (
												<span style={{ color: 'rgba(148, 163, 184, 0.3)' }}>—</span>
											)}
										</td>
															{metricColumns.map((metricKey) => {
											const metricValue = bucketValues[metricKey];
											const hasVal = typeof metricValue === 'number' && Number.isFinite(metricValue);
											return (
																	<td key={`${job.id}-${metricKey}`} style={{ padding: '0.75rem', textAlign: 'right' }}>
													{hasVal ? (
														<span style={{ fontFamily: 'monospace', color: '#e2e8f0' }}>
															{formatMetricValue(metricValue)}
														</span>
													) : (
														<span style={{ color: 'rgba(148, 163, 184, 0.3)' }}>—</span>
													)}
												</td>
											);
										})}
										<td style={{ padding: '0.75rem', color: 'rgba(148, 163, 184, 0.8)' }}>{bucketLabel}</td>
										<td style={{ padding: '0.75rem', textAlign: 'right', color: 'rgba(148, 163, 184, 0.8)' }}>{updatedAt ? formatRelativeTime(updatedAt) ?? updatedAt : '—'}</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			)}
		</section>
	);
};

export { ModelRegistrySection };
