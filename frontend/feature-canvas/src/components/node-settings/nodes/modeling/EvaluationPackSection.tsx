import React, {
	type ChangeEvent,
	type Dispatch,
	type SetStateAction,
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type {
	FeatureGraph,
	FeatureNodeParameter,
	ModelEvaluationConfusionMatrix,
	ModelEvaluationPrecisionRecallCurve,
	ModelEvaluationReport,
	ModelEvaluationResidualHistogram,
	ModelEvaluationResidualPoint,
	ModelEvaluationResiduals,
	ModelEvaluationRocCurve,
	ModelEvaluationSplitPayload,
	ModelEvaluationRequest,
	TrainingJobListResponse,
	TrainingJobResponse,
	TrainingJobSummary,
} from '../../../../api';
import {
	evaluateTrainingJob,
	fetchTrainingJob,
	fetchTrainingJobs,
	generatePipelineId,
} from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';
import { normalizeConfigBoolean } from '../../utils/configParsers';

type FormatMetricFn = (value?: number | null, precision?: number) => string;

const SPLIT_OPTIONS = [
	{ value: 'validation', label: 'Validation', accent: 'validation' },
	{ value: 'test', label: 'Test', accent: 'test' },
] as const;

type SplitValue = (typeof SPLIT_OPTIONS)[number]['value'];

const SPLIT_PRIORITY: Record<SplitValue, number> = {
	validation: 0,
	test: 1,
};

const DEFAULT_SPLITS: SplitValue[] = ['validation', 'test'];

const ensureStringArray = (value: unknown): string[] => {
	if (Array.isArray(value)) {
		return value
			.map((entry) => (typeof entry === 'string' ? entry : String(entry ?? '')))
			.map((entry) => entry.trim().toLowerCase())
			.filter(Boolean);
	}
	if (typeof value === 'string') {
		return value
			.split(',')
			.map((entry) => entry.trim().toLowerCase())
			.filter(Boolean);
	}
	return [];
};

const normalizeSplits = (splits: string[]): SplitValue[] => {
	const allowed = new Set(SPLIT_OPTIONS.map((option) => option.value));
	const next: SplitValue[] = [];
	splits.forEach((entry) => {
		if (!entry) {
			return;
		}
		// Skip train/training splits - not supported in evaluation
		if (entry === 'training' || entry === 'train') {
			return;
		}
		if (entry === 'valid' || entry === 'val') {
			next.push('validation');
			return;
		}
		if (allowed.has(entry as SplitValue)) {
			next.push(entry as SplitValue);
		}
	});
	const seen = new Set<SplitValue>();
	return next
		.filter((entry) => {
			if (seen.has(entry)) {
				return false;
			}
			seen.add(entry);
			return true;
		})
		.sort((a, b) => SPLIT_PRIORITY[a] - SPLIT_PRIORITY[b]);
};

	const arraysEqual = (left: string[], right: string[]): boolean => {
		if (left.length !== right.length) {
			return false;
		}
		for (let index = 0; index < left.length; index += 1) {
			if (left[index] !== right[index]) {
				return false;
			}
		}
		return true;
	};

const parseBooleanConfig = (value: unknown, fallback: boolean): boolean => {
	const normalized = normalizeConfigBoolean(value);
	if (normalized === null) {
		return fallback;
	}
	return normalized;
};

const parsePositiveIntegerConfig = (value: unknown, fallback: number): number => {
	if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
		return Math.round(value);
	}
	if (typeof value === 'string' && value.trim()) {
		const asNumber = Number(value);
		if (Number.isFinite(asNumber) && asNumber > 0) {
			return Math.round(asNumber);
		}
	}
	return fallback;
};

const toTimestamp = (value?: string | null): number => {
	if (!value) {
		return 0;
	}
	const parsed = Date.parse(value);
	return Number.isFinite(parsed) ? parsed : 0;
};

const friendlySplitLabel = (split: string): string => {
	const match = SPLIT_OPTIONS.find((option) => option.value === split);
	if (match) {
		return match.label;
	}
	return split.charAt(0).toUpperCase() + split.slice(1);
};

const statusLabel = (status: TrainingJobSummary['status']): string => {
	const formatted = status.replace(/_/g, ' ');
	return formatted.charAt(0).toUpperCase() + formatted.slice(1);
};

const buildJobLabel = (job: TrainingJobSummary): string => {
	const parts: string[] = [];
	if (job.model_type) {
		parts.push(job.model_type);
	}
	if (typeof job.version === 'number') {
		parts.push(`v${job.version}`);
	}
	parts.push(`· ${statusLabel(job.status)}`);
	const relative = formatRelativeTime(job.updated_at ?? job.created_at);
	if (relative) {
		parts.push(`(${relative})`);
	}
	return parts.join(' ');
};

const clamp01 = (value: number): number => {
	if (!Number.isFinite(value)) {
		return 0;
	}
	if (value < 0) {
		return 0;
	}
	if (value > 1) {
		return 1;
	}
	return value;
};

const extractNumericMetrics = (
	metrics: Record<string, unknown> | null | undefined
): Array<{ key: string; value: number }> => {
	if (!metrics || typeof metrics !== 'object') {
		return [];
	}
	const entries: Array<{ key: string; value: number }> = [];
	Object.entries(metrics).forEach(([key, value]) => {
		if (typeof value === 'number' && Number.isFinite(value)) {
			entries.push({ key, value });
		}
	});
	entries.sort((a, b) => a.key.localeCompare(b.key));
	return entries;
};

const formatPercentage = (value: number, formatMetricValue: FormatMetricFn): string => {
	if (!Number.isFinite(value)) {
		return '';
	}
	const resolved = formatMetricValue(value * 100, Math.abs(value) >= 0.1 ? 1 : 2);
	return resolved ? `${resolved}%` : '';
};

type EvaluationPackSectionProps = {
	nodeId: string;
	sourceId?: string | null;
	graph: FeatureGraph | null;
	config: Record<string, any> | null;
	setConfigState: Dispatch<SetStateAction<Record<string, any>>>;
	renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
	formatMetricValue: FormatMetricFn;
	connectedSplitHandles?: Partial<Record<SplitValue, boolean>>;
	canResetNode?: boolean;
	onResetNode?: () => void;
};

const EvaluationPackSection: React.FC<EvaluationPackSectionProps> = ({
	nodeId,
	sourceId,
	graph,
	config,
	setConfigState,
	renderParameterField: _renderParameterField,
	formatMetricValue,
	connectedSplitHandles,
	canResetNode = false,
	onResetNode,
}) => {
		void _renderParameterField;
	const queryClient = useQueryClient();

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
		graphPayload.nodes.forEach((item: any) => {
			if (item?.id) {
				nodeLookup.set(item.id, item);
			}
		});

		const direct = new Set<string>();
		graphPayload.edges.forEach((edge: any) => {
			if (!edge || edge.target !== nodeId) {
				return;
			}
			const sourceNode = nodeLookup.get(edge.source);
			if (sourceNode?.data?.catalogType === 'train_model_draft') {
				direct.add(sourceNode.id);
			}
		});

		if (direct.size > 0) {
			return Array.from(direct);
		}

		return graphPayload.nodes
			.filter((item: any) => item?.data?.catalogType === 'train_model_draft')
			.map((item: any) => item.id);
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
					setPipelineError(error?.message ?? 'Unable to compute pipeline signature.');
					setPipelineId(null);
				}
			});

		return () => {
			cancelled = true;
		};
	}, [graphPayload, sourceId]);

	const selectedJobId = useMemo(() => {
		const raw = config?.training_job_id;
		if (typeof raw === 'string') {
			return raw.trim();
		}
		if (typeof raw === 'number') {
			return String(raw);
		}
		return '';
	}, [config?.training_job_id]);

	const configSplits = useMemo(() => normalizeSplits(ensureStringArray(config?.splits)), [config?.splits]);
	const defaultSplitSelection = useMemo<SplitValue[]>(
		() => (configSplits.length ? configSplits : DEFAULT_SPLITS),
		[configSplits]
	);

	useEffect(() => {
		if (!connectedSplitHandles) {
			return;
		}
		setConfigState((prev) => {
			const previousConfig = prev ?? {};
			const existing = normalizeSplits(ensureStringArray(previousConfig.splits));
			const filtered = existing.filter((split) => connectedSplitHandles[split as SplitValue] !== false);
			if (
				filtered.length === existing.length &&
				filtered.every((value, index) => value === existing[index])
			) {
				return prev;
			}
			const nextConfig = { ...previousConfig } as Record<string, any>;
			if (filtered.length) {
				nextConfig.splits = filtered;
			} else {
				delete nextConfig.splits;
			}
			return nextConfig;
		});
	}, [connectedSplitHandles, setConfigState]);

	const selectedSplits = useMemo<SplitValue[]>(() => {
		if (!connectedSplitHandles) {
			return defaultSplitSelection;
		}
		return defaultSplitSelection.filter((split) => connectedSplitHandles[split] !== false);
	}, [connectedSplitHandles, defaultSplitSelection]);

	const missingSplitLabels = useMemo(() => {
		if (!connectedSplitHandles) {
			return [] as string[];
		}
		return SPLIT_OPTIONS.filter((option) => connectedSplitHandles[option.value] === false).map(
			(option) => option.label
		);
	}, [connectedSplitHandles]);

	const missingSplitSummary = useMemo(() => {
		if (!missingSplitLabels.length) {
			return '';
		}
		if (missingSplitLabels.length === 1) {
			return missingSplitLabels[0];
		}
		return `${missingSplitLabels[0]} and ${missingSplitLabels[1]}`;
	}, [missingSplitLabels]);

	const hasSelectableSplits = selectedSplits.length > 0;

	const includeConfusion = parseBooleanConfig(config?.include_confusion, true);
	const includeCurves = parseBooleanConfig(config?.include_curves, true);
	const includeResiduals = parseBooleanConfig(config?.include_residuals, true);
	const maxCurvePoints = parsePositiveIntegerConfig(config?.max_curve_points, 500);
	const maxScatterPoints = parsePositiveIntegerConfig(config?.max_scatter_points, 750);
	const lastEvaluatedAt = typeof config?.last_evaluated_at === 'string' ? config.last_evaluated_at : null;

	const [report, setReport] = useState<ModelEvaluationReport | null>(null);
	const userModifiedSplitsRef = useRef(false);

	useEffect(() => {
		// Reset the current report when switching jobs.
		setReport(null);
		userModifiedSplitsRef.current = false;
	}, [selectedJobId]);

	const trainingJobsQueryKey = useMemo(
		() => [
			'feature-canvas',
			'model-evaluation',
			sourceId ?? 'none',
			pipelineId ?? 'pending',
			upstreamKey,
		],
		[pipelineId, sourceId, upstreamKey]
	);

	const trainingJobsQuery = useQuery<TrainingJobListResponse, Error>({
		queryKey: trainingJobsQueryKey,
		queryFn: async () => {
			const baseParams = {
				datasetSourceId: sourceId || undefined,
				limit: 50,
			};

			const aggregated = new Map<string, TrainingJobSummary>();
			const encounteredErrors: Error[] = [];
			let datasetResponse: TrainingJobListResponse | null = null;

			const appendJobs = (jobs?: TrainingJobSummary[] | null) => {
				if (!Array.isArray(jobs)) {
					return;
				}
				jobs.forEach((job) => {
					if (job && !aggregated.has(job.id)) {
						aggregated.set(job.id, job);
					}
				});
			};

			const safeFetch = async (params: {
				datasetSourceId?: string;
				pipelineId?: string;
				nodeId?: string;
				limit?: number;
			}) => {
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

			if (upstreamTrainingNodeIds.length) {
				await Promise.all(
					upstreamTrainingNodeIds.map((node) => safeFetch({ ...baseParams, nodeId: node }))
				);
			}

			datasetResponse = await safeFetch(baseParams);

			if (!aggregated.size) {
				if (datasetResponse) {
					return datasetResponse;
				}
				if (encounteredErrors.length) {
					throw encounteredErrors[0];
				}
				return { jobs: [], total: 0 };
			}

			const jobs = Array.from(aggregated.values());
			jobs.sort((a, b) => toTimestamp(b.updated_at ?? b.created_at) - toTimestamp(a.updated_at ?? a.created_at));
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
			const current = query.state.data as TrainingJobListResponse | undefined;
			const jobs = current?.jobs ?? [];
			const hasActiveJob = jobs.some((job) => job.status === 'queued' || job.status === 'running');
			return hasActiveJob ? 5000 : false;
		},
	});

	const jobList = trainingJobsQuery.data?.jobs ?? [];
	const jobLookup = useMemo(() => {
		const map = new Map<string, TrainingJobSummary>();
		jobList.forEach((job) => {
			map.set(job.id, job);
		});
		return map;
	}, [jobList]);

	const successfulJobs = useMemo(() => jobList.filter((job) => job.status === 'succeeded'), [jobList]);

	const jobOptions = useMemo(
		() =>
			successfulJobs.map((job) => ({ value: job.id, label: buildJobLabel(job) })),
		[successfulJobs]
	);

	const selectedJob = selectedJobId ? jobLookup.get(selectedJobId) ?? null : null;

	const jobDetailQuery = useQuery<TrainingJobResponse, Error>({
		queryKey: ['feature-canvas', 'training-job', selectedJobId || 'none'],
		queryFn: () => fetchTrainingJob(selectedJobId),
		enabled: Boolean(selectedJobId),
		staleTime: 30 * 1000,
	});

		useEffect(() => {
		const job = jobDetailQuery.data;
		if (!job) {
			return;
		}
		const evaluation = job.metrics?.evaluation;
		if (!evaluation || typeof evaluation !== 'object') {
			return;
		}
		const rawSplits = evaluation.splits;
		if (!rawSplits || typeof rawSplits !== 'object') {
			return;
		}

		const normalizedSplits: Record<string, ModelEvaluationSplitPayload> = {};
		Object.entries(rawSplits).forEach(([key, value]) => {
			if (value && typeof value === 'object') {
				normalizedSplits[key] = value as ModelEvaluationSplitPayload;
			}
		});

		const generatedAt = typeof evaluation.generated_at === 'string'
			? evaluation.generated_at
			: lastEvaluatedAt ?? job.updated_at ?? job.created_at ?? new Date().toISOString();

		const problemType = evaluation.problem_type === 'regression' ? 'regression' : 'classification';
			const hydratedReport: ModelEvaluationReport = {
			job_id: job.id,
			pipeline_id: job.pipeline_id,
			node_id: job.node_id,
			generated_at: generatedAt,
			problem_type: problemType,
			target_column: evaluation.target_column ?? job.metadata?.target_column ?? null,
			splits: normalizedSplits,
		};
		setReport(hydratedReport);

		if (generatedAt && generatedAt !== lastEvaluatedAt) {
			setConfigState((prev) => ({ ...(prev ?? {}), last_evaluated_at: generatedAt }));
		}

			const splitKeys = normalizeSplits(Object.keys(normalizedSplits));
			// Only auto-update splits if the user hasn't manually modified them
			if (splitKeys.length && !userModifiedSplitsRef.current && !arraysEqual(configSplits, splitKeys)) {
				setConfigState((prev) => ({ ...(prev ?? {}), splits: splitKeys }));
			}
		}, [jobDetailQuery.data, lastEvaluatedAt, setConfigState, configSplits]);

	const evaluationMutation = useMutation<ModelEvaluationReport, Error, ModelEvaluationRequest>({
		mutationFn: async (payload) => {
			if (!selectedJobId) {
				throw new Error('Select a training job before running evaluation.');
			}
			return evaluateTrainingJob(selectedJobId, payload);
		},
		onSuccess: (data) => {
			setReport(data);
			setConfigState((prev) => ({
				...(prev ?? {}),
				training_job_id: selectedJobId,
				splits: data.splits ? Object.keys(data.splits) : selectedSplits,
				include_confusion: includeConfusion,
				include_curves: includeCurves,
				include_residuals: includeResiduals,
				max_curve_points: maxCurvePoints,
				max_scatter_points: maxScatterPoints,
				last_evaluated_at: data.generated_at,
			}));

			queryClient.invalidateQueries({ queryKey: ['feature-canvas', 'training-job', selectedJobId] });
		},
	});

	const handleSelectJob = useCallback(
		(event: ChangeEvent<HTMLSelectElement>) => {
			const value = event.target.value;
			evaluationMutation.reset();
			setConfigState((prev) => {
				const next = { ...(prev ?? {}) } as Record<string, any>;
				if (value) {
					next.training_job_id = value;
				} else {
					delete next.training_job_id;
				}
				return next;
			});
		},
		[evaluationMutation, setConfigState]
	);

	const handleToggleSplit = useCallback(
		(split: SplitValue) => {
			if (connectedSplitHandles && connectedSplitHandles[split] === false) {
				return;
			}
			evaluationMutation.reset();
			userModifiedSplitsRef.current = true;
			setConfigState((prev) => {
				const base = normalizeSplits(ensureStringArray(prev?.splits));
				const hasSplit = base.includes(split);
				let nextSplits: SplitValue[];
				if (hasSplit) {
					if (base.length <= 1) {
						return prev ?? {};
					}
					nextSplits = base.filter((entry) => entry !== split);
				} else {
					nextSplits = normalizeSplits([...base, split]);
				}
				return { ...(prev ?? {}), splits: nextSplits };
			});
			},
			[connectedSplitHandles, evaluationMutation, setConfigState]
	);

	const handleToggleInclude = useCallback(
		(key: 'include_confusion' | 'include_curves' | 'include_residuals', value: boolean) => {
			evaluationMutation.reset();
			setConfigState((prev) => ({ ...(prev ?? {}), [key]: value }));
		},
		[evaluationMutation, setConfigState]
	);

	const handleRunEvaluation = useCallback(() => {
		if (!selectedJobId) {
			return;
		}
		const request: ModelEvaluationRequest = {
			splits: selectedSplits,
			include_confusion: includeConfusion,
			include_curves: includeCurves,
			include_residuals: includeResiduals,
			max_curve_points: includeCurves ? maxCurvePoints : undefined,
			max_scatter_points: includeResiduals ? maxScatterPoints : undefined,
		};
		evaluationMutation.mutate(request);
	}, [evaluationMutation, includeConfusion, includeCurves, includeResiduals, maxCurvePoints, maxScatterPoints, selectedJobId, selectedSplits]);

	const lastRunRelative = report ? formatRelativeTime(report.generated_at) : lastEvaluatedAt ? formatRelativeTime(lastEvaluatedAt) : null;

	const orderedSplitEntries = useMemo(() => {
		if (!report) {
			return [] as Array<[string, ModelEvaluationSplitPayload]>;
		}
		return Object.entries(report.splits).sort((a, b) => {
			const [keyA] = a;
			const [keyB] = b;
			const priorityA = SPLIT_PRIORITY[keyA as SplitValue] ?? 99;
			const priorityB = SPLIT_PRIORITY[keyB as SplitValue] ?? 99;
			return priorityA - priorityB;
		});
	}, [report]);

	return (
		<div className="node-settings__section node-settings__evaluation">
			<div className="node-settings__section-header">
				<h3>Model Evaluation</h3>
				<div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
					{lastRunRelative && <span className="evaluation-status">Last ran {lastRunRelative}</span>}
					{canResetNode && (
						<button
							type="button"
							className="btn btn-outline-secondary btn-sm"
							onClick={onResetNode}
							disabled={!onResetNode}
							title="Reset evaluation node to defaults"
						>
							Reset node
						</button>
					)}
				</div>
			</div>

			<div className="evaluation-controls">
				<label className="evaluation-control">
					<span className="evaluation-control__label">Training job</span>
					<select
						className="evaluation-control__select"
						value={selectedJobId}
						onChange={handleSelectJob}
						disabled={trainingJobsQuery.isLoading || trainingJobsQuery.isFetching}
					>
						<option value="">Select a trained model…</option>
						{jobOptions.map((option) => (
							<option key={option.value} value={option.value}>
								{option.label}
							</option>
						))}
					</select>
					{trainingJobsQuery.isLoading && (
						<span className="evaluation-hint">Loading training jobs…</span>
					)}
					{trainingJobsQuery.error && (
						<span className="evaluation-hint evaluation-hint--error">
							{trainingJobsQuery.error.message}
						</span>
					)}
					{!trainingJobsQuery.isLoading && !jobOptions.length && (
						<span className="evaluation-hint">No successful training jobs detected yet.</span>
					)}
					{pipelineError && (
						<span className="evaluation-hint evaluation-hint--warning">{pipelineError}</span>
					)}
				</label>

				<div className="evaluation-control">
					<span className="evaluation-control__label">Dataset splits</span>
					<div className="evaluation-split-toggle">
						{SPLIT_OPTIONS.map((option) => {
							const isConnected = connectedSplitHandles ? connectedSplitHandles[option.value] !== false : true;
							const isActive = isConnected && selectedSplits.includes(option.value);
							return (
								<label
									key={option.value}
									className={`evaluation-chip evaluation-chip--${option.accent} ${isActive ? 'evaluation-chip--active' : ''}`}
									aria-disabled={!isConnected}
									title={!isConnected ? `${option.label} split is not connected` : undefined}
									style={
										!isConnected
											? {
												opacity: 0.45,
												cursor: 'not-allowed',
												borderStyle: 'dashed',
											}
											: undefined
									}
								>
									<input
										type="checkbox"
										checked={isActive}
										disabled={!isConnected}
										onChange={() => handleToggleSplit(option.value)}
									/>
									<span>{option.label}</span>
								</label>
							);
						})}
					</div>
					{missingSplitLabels.length > 0 && (
						<span className="evaluation-hint evaluation-hint--warning">
							Connect {missingSplitSummary} split{missingSplitLabels.length > 1 ? 's' : ''} to enable diagnostics.
						</span>
					)}
					{!hasSelectableSplits && (
						<span className="evaluation-hint evaluation-hint--warning">
							No dataset splits are available. Attach a validation or test split to run evaluation.
						</span>
					)}
				</div>

				<div className="evaluation-control evaluation-control--toggles">
					<label className="evaluation-toggle">
						<input
							type="checkbox"
							checked={includeConfusion}
							onChange={(event) => handleToggleInclude('include_confusion', event.target.checked)}
						/>
						<span>Confusion matrix</span>
					</label>
					<label className="evaluation-toggle">
						<input
							type="checkbox"
							checked={includeCurves}
							onChange={(event) => handleToggleInclude('include_curves', event.target.checked)}
						/>
						<span>ROC / PR curves</span>
					</label>
					<label className="evaluation-toggle">
						<input
							type="checkbox"
							checked={includeResiduals}
							onChange={(event) => handleToggleInclude('include_residuals', event.target.checked)}
						/>
						<span>Residual diagnostics</span>
					</label>
				</div>
			</div>

			<div className="evaluation-actions">
				<button
					type="button"
					className="evaluation-button"
					onClick={handleRunEvaluation}
					disabled={!selectedJobId || evaluationMutation.isPending || !hasSelectableSplits}
				>
					{evaluationMutation.isPending ? 'Running…' : 'Run evaluation'}
				</button>
				{selectedJob && selectedJob.status !== 'succeeded' && (
					<span className="evaluation-hint evaluation-hint--warning">
						Selected job is not marked as succeeded; results may be incomplete.
					</span>
				)}
				{!hasSelectableSplits && (
					<span className="evaluation-hint evaluation-hint--warning">
						Connect at least one dataset split to run evaluation diagnostics.
					</span>
				)}
				{evaluationMutation.isError && (
					<span className="evaluation-hint evaluation-hint--error">
						{evaluationMutation.error.message}
					</span>
				)}
				{evaluationMutation.isSuccess && report && (
					<span className="evaluation-hint evaluation-hint--success">
						Diagnostics generated {formatRelativeTime(report.generated_at)}.
					</span>
				)}
				{jobDetailQuery.error && (
					<span className="evaluation-hint evaluation-hint--error">
						{jobDetailQuery.error.message}
					</span>
				)}
			</div>

			{report ? (
				<div className="node-settings__evaluation-results">
					<div className="evaluation-summary">
						<div>
							<span className="evaluation-summary__label">Problem type</span>
							<span className="evaluation-summary__value">
								{report.problem_type === 'regression' ? 'Regression' : 'Classification'}
							</span>
						</div>
						{report.target_column && (
							<div>
								<span className="evaluation-summary__label">Target column</span>
								<span className="evaluation-summary__value">{report.target_column}</span>
							</div>
						)}
						<div>
							<span className="evaluation-summary__label">Evaluation time</span>
							<span className="evaluation-summary__value">
								{formatRelativeTime(report.generated_at) ?? report.generated_at}
							</span>
						</div>
					</div>

					{orderedSplitEntries.map(([splitKey, splitPayload]) => (
						<EvaluationSplitCard
							key={splitKey}
							splitKey={splitKey}
							payload={splitPayload}
							problemType={report.problem_type}
							formatMetricValue={formatMetricValue}
						/>
					))}
				</div>
			) : (
				<div className="evaluation-placeholder">
					{selectedJobId ? (
						<p>Run the evaluation to generate metrics and diagnostics for the selected model.</p>
					) : (
						<p>Select a completed training job to enable the evaluation workflow.</p>
					)}
				</div>
			)}
		</div>
	);
};

type EvaluationSplitCardProps = {
	splitKey: string;
	payload: ModelEvaluationSplitPayload;
	problemType: 'classification' | 'regression';
	formatMetricValue: FormatMetricFn;
};

const EvaluationSplitCard: React.FC<EvaluationSplitCardProps> = ({
	splitKey,
	payload,
	problemType,
	formatMetricValue,
}) => {
	const metricEntries = extractNumericMetrics(payload.metrics);
	const hasMetrics = metricEntries.length > 0;
	const notes = Array.isArray(payload.notes) ? payload.notes.filter(Boolean) : [];

	return (
		<section className="evaluation-split">
			<header className="evaluation-split__header">
				<div>
					<h4>{friendlySplitLabel(splitKey)} split</h4>
					<span className="evaluation-split__rows">{payload.row_count} rows</span>
				</div>
			</header>
			<div className="evaluation-split__content">
				<div className="evaluation-metrics">
					<h5>Metrics</h5>
					{hasMetrics ? (
						<div className="evaluation-metrics__grid">
							{metricEntries.map((entry) => (
								<div key={entry.key} className="evaluation-metrics__item">
									<span className="evaluation-metrics__label">{entry.key}</span>
									<span className="evaluation-metrics__value">{formatMetricValue(entry.value)}</span>
								</div>
							))}
						</div>
					) : (
						<p className="evaluation-hint">No numeric metrics reported for this split.</p>
					)}
				</div>

				{problemType === 'classification' && payload.confusion_matrix && (
					<ConfusionMatrixTable
						matrix={payload.confusion_matrix}
						formatMetricValue={formatMetricValue}
					/>
				)}

				{problemType === 'classification' && payload.roc_curves.length > 0 && (
					<div className="evaluation-curves">
						{payload.roc_curves.map((curve) => (
							<RocCurveChart
								key={`roc-${curve.label}`}
								curve={curve}
								formatMetricValue={formatMetricValue}
							/>
						))}
						{payload.pr_curves.map((curve) => (
							<PrecisionRecallCurveChart
								key={`pr-${curve.label}`}
								curve={curve}
								formatMetricValue={formatMetricValue}
							/>
						))}
					</div>
				)}

				{problemType === 'regression' && payload.residuals && (
					<RegressionDiagnostics
						residuals={payload.residuals}
						formatMetricValue={formatMetricValue}
					/>
				)}
			</div>
			{notes.length > 0 && (
				<ul className="evaluation-notes">
					{notes.map((note, index) => (
						<li key={`${splitKey}-note-${index}`}>{note}</li>
					))}
				</ul>
			)}
		</section>
	);
};

type ConfusionMatrixTableProps = {
	matrix: ModelEvaluationConfusionMatrix;
	formatMetricValue: FormatMetricFn;
};

const ConfusionMatrixTable: React.FC<ConfusionMatrixTableProps> = ({ matrix, formatMetricValue }) => {
	const labels = Array.isArray(matrix.labels) ? matrix.labels : [];
	const totals = Array.isArray(matrix.totals) ? matrix.totals : [];
	const normalized = Array.isArray(matrix.normalized) ? matrix.normalized : [];
	const accuracy = typeof matrix.accuracy === 'number' ? matrix.accuracy : null;

	return (
		<div className="evaluation-confusion">
			<h5>Confusion matrix</h5>
			{accuracy !== null && (
				<span className="evaluation-confusion__accuracy">
					Accuracy {formatPercentage(accuracy, formatMetricValue)}
				</span>
			)}
			<table>
				<thead>
					<tr>
						<th>Actual \ Predicted</th>
						{labels.map((label, index) => (
							<th key={`pred-${index}`}>{label}</th>
						))}
						<th>Total</th>
					</tr>
				</thead>
				<tbody>
					{matrix.matrix.map((row, rowIndex) => {
						const normalizedRow = normalized[rowIndex] ?? [];
						const total = totals[rowIndex] ?? row.reduce((acc, value) => acc + value, 0);
						return (
							<tr key={`row-${rowIndex}`}>
								<th>{labels[rowIndex] ?? `Class ${rowIndex}`}</th>
								{row.map((value, columnIndex) => {
									const share = typeof normalizedRow[columnIndex] === 'number' ? normalizedRow[columnIndex] : null;
									return (
										<td key={`cell-${rowIndex}-${columnIndex}`}>
											<span className="evaluation-confusion__count">{value}</span>
											{share !== null && (
												<span className="evaluation-confusion__percentage">
													{formatPercentage(share, formatMetricValue)}
												</span>
											)}
										</td>
									);
								})}
								<td>{total}</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
};

type RocCurveChartProps = {
	curve: ModelEvaluationRocCurve;
	formatMetricValue: FormatMetricFn;
};

const RocCurveChart: React.FC<RocCurveChartProps> = ({ curve, formatMetricValue }) => {
	if (!Array.isArray(curve.fpr) || !Array.isArray(curve.tpr) || !curve.fpr.length || curve.fpr.length !== curve.tpr.length) {
		return null;
	}

	const width = 240;
	const height = 240;
	const margin = 28;
	const span = width - margin * 2;
	const verticalSpan = height - margin * 2;

	const path = curve.fpr
		.map((value, index) => {
			const x = margin + clamp01(value) * span;
			const y = height - margin - clamp01(curve.tpr[index]) * verticalSpan;
			return `${index === 0 ? 'M' : 'L'}${x} ${y}`;
		})
		.join(' ');

	return (
		<div className="evaluation-chart">
			<svg
				viewBox={`0 0 ${width} ${height}`}
				role="img"
				aria-label={`ROC curve for ${curve.label}`}
			>
				<rect x={margin} y={margin} width={span} height={verticalSpan} className="evaluation-chart__frame" />
				<line x1={margin} y1={height - margin} x2={width - margin} y2={margin} className="evaluation-curve__baseline" />
				<line x1={margin} y1={margin} x2={margin} y2={height - margin} className="evaluation-chart__axis" />
				<line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} className="evaluation-chart__axis" />
				<path d={path} className="evaluation-curve__path" />
				<text x={width / 2} y={height - 4} className="evaluation-chart__tick" textAnchor="middle">
					False positive rate
				</text>
				<text
					x={-(height / 2)}
					y={12}
					transform="rotate(-90)"
					className="evaluation-chart__tick"
					textAnchor="middle"
				>
					True positive rate
				</text>
			</svg>
			<div className="evaluation-chart__caption">
				<span>{curve.label}</span>
				{typeof curve.auc === 'number' && (
					<span>AUC {formatMetricValue(curve.auc)}</span>
				)}
			</div>
		</div>
	);
};

type PrecisionRecallCurveChartProps = {
	curve: ModelEvaluationPrecisionRecallCurve;
	formatMetricValue: FormatMetricFn;
};

const PrecisionRecallCurveChart: React.FC<PrecisionRecallCurveChartProps> = ({ curve, formatMetricValue }) => {
	if (!Array.isArray(curve.recall) || !Array.isArray(curve.precision) || !curve.recall.length || curve.recall.length !== curve.precision.length) {
		return null;
	}

	const width = 240;
	const height = 240;
	const margin = 28;
	const span = width - margin * 2;
	const verticalSpan = height - margin * 2;

	const path = curve.recall
		.map((value, index) => {
			const x = margin + clamp01(value) * span;
			const y = height - margin - clamp01(curve.precision[index]) * verticalSpan;
			return `${index === 0 ? 'M' : 'L'}${x} ${y}`;
		})
		.join(' ');

	return (
		<div className="evaluation-chart">
			<svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Precision/recall for ${curve.label}`}>
				<rect x={margin} y={margin} width={span} height={verticalSpan} className="evaluation-chart__frame" />
				<line x1={margin} y1={margin} x2={margin} y2={height - margin} className="evaluation-chart__axis" />
				<line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} className="evaluation-chart__axis" />
				<path d={path} className="evaluation-curve__path" />
				<text x={width / 2} y={height - 4} className="evaluation-chart__tick" textAnchor="middle">
					Recall
				</text>
				<text
					x={-(height / 2)}
					y={12}
					transform="rotate(-90)"
					className="evaluation-chart__tick"
					textAnchor="middle"
				>
					Precision
				</text>
			</svg>
			<div className="evaluation-chart__caption">
				<span>{curve.label}</span>
				{typeof curve.average_precision === 'number' && (
					<span>AP {formatMetricValue(curve.average_precision)}</span>
				)}
			</div>
		</div>
	);
};

type RegressionDiagnosticsProps = {
	residuals: ModelEvaluationResiduals;
	formatMetricValue: FormatMetricFn;
};

const RESIDUAL_SUMMARY_LABELS: Record<string, string> = {
	residual_min: 'Residual min',
	residual_max: 'Residual max',
	residual_mean: 'Residual mean',
	residual_std: 'Residual std',
};

const RegressionDiagnostics: React.FC<RegressionDiagnosticsProps> = ({ residuals, formatMetricValue }) => {
	const summaryEntries = Object.entries(residuals.summary ?? {}).filter((entry): entry is [string, number] => {
		return typeof entry[1] === 'number' && Number.isFinite(entry[1]);
	});

	return (
		<div className="evaluation-regression">
			<h5>Residual analysis</h5>
			{summaryEntries.length > 0 && (
				<div className="evaluation-summary evaluation-summary--compact">
					{summaryEntries.map(([key, value]) => (
						<div key={key}>
							<span className="evaluation-summary__label">{RESIDUAL_SUMMARY_LABELS[key] ?? key}</span>
							<span className="evaluation-summary__value">{formatMetricValue(value)}</span>
						</div>
					))}
				</div>
			)}
			<ResidualHistogramChart histogram={residuals.histogram} />
			{residuals.scatter && residuals.scatter.length > 0 && (
				<ResidualScatterPlot points={residuals.scatter} />
			)}
		</div>
	);
};

type ResidualHistogramChartProps = {
	histogram: ModelEvaluationResidualHistogram;
};

const ResidualHistogramChart: React.FC<ResidualHistogramChartProps> = ({ histogram }) => {
	const counts = Array.isArray(histogram.counts) ? histogram.counts : [];
	const binEdges = Array.isArray(histogram.bin_edges) ? histogram.bin_edges : [];
	if (!counts.length || binEdges.length !== counts.length + 1) {
		return null;
	}
	const maxCount = counts.reduce((acc, value) => Math.max(acc, value), 0) || 1;
	return (
		<div className="evaluation-histogram" aria-label="Residual histogram">
			{counts.map((count, index) => {
				const height = (count / maxCount) * 100;
				return (
					<div key={`bin-${index}`} className="evaluation-histogram__bin" title={`${count} rows`}>
						<div className="evaluation-histogram__bar" style={{ height: `${height}%` }} />
					</div>
				);
			})}
		</div>
	);
};

type ResidualScatterPlotProps = {
	points: ModelEvaluationResidualPoint[];
};

const ResidualScatterPlot: React.FC<ResidualScatterPlotProps> = ({ points }) => {
	if (!Array.isArray(points) || !points.length) {
		return null;
	}

	const width = 320;
	const height = 260;
	const margin = 36;

	const actualValues = points.map((point) => point.actual);
	const predictedValues = points.map((point) => point.predicted);
	const globalMin = Math.min(...actualValues, ...predictedValues);
	const globalMax = Math.max(...actualValues, ...predictedValues);
	const span = globalMax - globalMin || 1;

	const scale = (value: number) => (value - globalMin) / span;
	const projectX = (value: number) => margin + scale(value) * (width - margin * 2);
	const projectY = (value: number) => height - margin - scale(value) * (height - margin * 2);

	return (
		<div className="evaluation-scatter" aria-label="Actual vs predicted scatter">
			<svg viewBox={`0 0 ${width} ${height}`}>
				<rect x={margin} y={margin} width={width - margin * 2} height={height - margin * 2} className="evaluation-chart__frame" />
				<line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} className="evaluation-chart__axis" />
				<line x1={margin} y1={margin} x2={margin} y2={height - margin} className="evaluation-chart__axis" />
				<line
					x1={projectX(globalMin)}
					y1={projectY(globalMin)}
					x2={projectX(globalMax)}
					y2={projectY(globalMax)}
					className="evaluation-curve__baseline"
				/>
				{points.map((point, index) => (
					<circle
						key={`scatter-${index}`}
						cx={projectX(point.actual)}
						cy={projectY(point.predicted)}
						r={2.1}
						className="evaluation-scatter__point"
					/>
				))}
			</svg>
		</div>
	);
};

export { EvaluationPackSection };
