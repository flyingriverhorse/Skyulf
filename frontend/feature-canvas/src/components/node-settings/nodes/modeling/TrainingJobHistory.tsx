import React, { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { TrainingJobResponse, TrainingJobSummary } from '../../../../api';
import { cancelTrainingJob } from '../../../../api';
import { formatRelativeTime } from '../../utils/formatters';
import { STATUS_LABEL, pickPrimaryMetric, formatMetricValue } from './modelingUtils';
import { ModelComparisonTable } from './ModelComparisonTable';

type TrainingJobHistoryProps = {
	jobs: TrainingJobSummary[];
	isLoading: boolean;
	hiddenJobCount: number;
	onRefresh: () => void;
	pipelineId: string | null;
	createJobError: Error | null;
	jobsError: Error | null;
	tuningJobsError: Error | null;
	lastCreatedJob: TrainingJobResponse | null;
	lastCreatedJobCount: number;
	isTuningJobsLoading: boolean;
};

export const TrainingJobHistory: React.FC<TrainingJobHistoryProps> = ({
	jobs,
	isLoading,
	hiddenJobCount,
	onRefresh,
	pipelineId,
	createJobError,
	jobsError,
	tuningJobsError,
	lastCreatedJob,
	lastCreatedJobCount,
	isTuningJobsLoading,
}) => {
	const [viewMode, setViewMode] = useState<'list' | 'compare'>('list');
	const queryClient = useQueryClient();

	const cancelMutation = useMutation({
		mutationFn: cancelTrainingJob,
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ['training-jobs'] });
		},
		onError: (error) => {
			console.error('Failed to cancel job:', error);
			alert(error instanceof Error ? error.message : 'Failed to cancel job');
		},
	});

	const handleCancel = (jobId: string) => {
		if (confirm('Are you sure you want to cancel this training job?')) {
			cancelMutation.mutate(jobId);
		}
	};

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
		const jobError = typeof job.error_message === 'string' ? job.error_message.trim() : '';

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
							<strong>v{job.version}</strong> — {statusLabel}
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
					{detailParts.length > 0 ? detailParts.join(' • ') : ''}
				</div>
			</li>
		);
	};

	return (
		<section className="canvas-modal__section">
			<div className="canvas-modal__section-header">
				<h3>Background training jobs</h3>
				<div className="canvas-modal__section-actions">
					<div style={{ display: 'flex', gap: '0.25rem', marginRight: '0.75rem' }}>
						<button
							type="button"
							className={`btn ${viewMode === 'list' ? 'btn-secondary' : 'btn-outline-secondary'}`}
							onClick={() => setViewMode('list')}
							style={{ padding: '0.25rem 0.75rem', fontSize: '0.8rem' }}
						>
							List
						</button>
						<button
							type="button"
							className={`btn ${viewMode === 'compare' ? 'btn-secondary' : 'btn-outline-secondary'}`}
							onClick={() => setViewMode('compare')}
							style={{ padding: '0.25rem 0.75rem', fontSize: '0.8rem' }}
						>
							Compare
						</button>
					</div>
					<button
						type="button"
						className="btn btn-outline-secondary"
						onClick={onRefresh}
						disabled={isLoading}
					>
						{isLoading ? 'Refreshing…' : 'Refresh jobs'}
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

			{isLoading && <p className="canvas-modal__note canvas-modal__note--muted">Loading recent jobs…</p>}
			{hiddenJobCount > 0 && (
				<p className="canvas-modal__note canvas-modal__note--muted">
					{hiddenJobCount === 1
						? '1 training job uses a different problem type. Switch the problem type to view it.'
						: `${hiddenJobCount} training jobs use a different problem type. Switch the problem type to view them.`}
				</p>
			)}
			{!isLoading && jobs.length === 0 && (
				<p className="canvas-modal__note canvas-modal__note--muted">No training jobs found for this node.</p>
			)}
			{jobs.length > 0 && (
				viewMode === 'list' ? (
					<ul className="canvas-modal__note-list">{jobs.map(renderJobSummary)}</ul>
				) : (
					<ModelComparisonTable jobs={jobs} />
				)
			)}
		</section>
	);
};
