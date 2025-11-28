import React, { useMemo, useState } from 'react';
import type { TrainingJobSummary } from '../../../../api';
import { formatMetricValue } from './modelingUtils';
import { formatRelativeTime } from '../../utils/formatters';

type SortConfig = {
	key: string;
	direction: 'asc' | 'desc';
};

type ModelComparisonTableProps = {
	jobs: TrainingJobSummary[];
};

export const ModelComparisonTable: React.FC<ModelComparisonTableProps> = ({ jobs }) => {
	const [sortConfig, setSortConfig] = useState<SortConfig>({ key: 'created_at', direction: 'desc' });

	// Identify problem type from the jobs to decide which columns to show
	const problemType = useMemo(() => {
		const types = jobs
			.map((j) => j.problem_type || j.metadata?.problem_type || j.metadata?.resolved_problem_type)
			.filter(Boolean);
		// Simple heuristic: if any say regression, treat as regression, else classification
		return types.some((t) => String(t).toLowerCase() === 'regression') ? 'regression' : 'classification';
	}, [jobs]);

	const metricColumns = useMemo(() => {
		let columns = [];
		if (problemType === 'regression') {
			columns = [
				{ key: 'r2', label: 'R²' },
				{ key: 'rmse', label: 'RMSE' },
				{ key: 'mae', label: 'MAE' },
				{ key: 'mse', label: 'MSE' },
			];
		} else {
			columns = [
				{ key: 'accuracy', label: 'Accuracy' },
				{ key: 'f1', label: 'F1' },
				{ key: 'f1_weighted', label: 'F1 (Weighted)' },
				{ key: 'precision', label: 'Precision' },
				{ key: 'precision_weighted', label: 'Precision (Weighted)' },
				{ key: 'recall', label: 'Recall' },
				{ key: 'recall_weighted', label: 'Recall (Weighted)' },
				{ key: 'roc_auc', label: 'AUC' },
				{ key: 'roc_auc_weighted', label: 'AUC (Weighted)' },
			];
		}

		// Filter columns based on actual data presence
		const availableColumns = columns.filter((col) => {
			return jobs.some((job) => {
				const metrics = job.metrics || {};
				// Helper to find metric value in nested structure (test > validation > train)
				const findMetricValue = (key: string) => {
					// 1. Try direct access (legacy)
					if (metrics[key] !== undefined) return metrics[key];

					// 2. Try nested buckets (test -> validation -> train)
					const buckets = [metrics.test, metrics.validation, metrics.train];
					for (const bucket of buckets) {
						if (bucket && typeof bucket === 'object' && bucket[key] !== undefined) {
							return bucket[key];
						}
					}

					// 3. Try cross-validation mean
					if (metrics.cross_validation?.metrics?.mean?.[key] !== undefined) {
						return metrics.cross_validation.metrics.mean[key];
					}

					return undefined;
				};

				return findMetricValue(col.key) !== undefined;
			});
		});

		// Hide weighted metrics if unweighted versions are present for all jobs that have the weighted one
		// (i.e., if we have binary classification jobs, we prefer the unweighted metric and hide the weighted one to avoid clutter)
		const hasUnweightedF1 = availableColumns.some(c => c.key === 'f1');
		const hasUnweightedPrecision = availableColumns.some(c => c.key === 'precision');
		const hasUnweightedRecall = availableColumns.some(c => c.key === 'recall');

		return availableColumns.filter(col => {
			if (col.key === 'f1_weighted' && hasUnweightedF1) {
				// Only hide if every job with f1_weighted ALSO has f1 (meaning no multiclass-only jobs)
				const hasWeightedOnly = jobs.some(job => {
					const metrics = job.metrics || {};
					const hasW = metrics.f1_weighted !== undefined || metrics.test?.f1_weighted !== undefined;
					const hasU = metrics.f1 !== undefined || metrics.test?.f1 !== undefined;
					return hasW && !hasU;
				});
				if (!hasWeightedOnly) return false;
			}
			if (col.key === 'precision_weighted' && hasUnweightedPrecision) {
				const hasWeightedOnly = jobs.some(job => {
					const metrics = job.metrics || {};
					const hasW = metrics.precision_weighted !== undefined || metrics.test?.precision_weighted !== undefined;
					const hasU = metrics.precision !== undefined || metrics.test?.precision !== undefined;
					return hasW && !hasU;
				});
				if (!hasWeightedOnly) return false;
			}
			if (col.key === 'recall_weighted' && hasUnweightedRecall) {
				const hasWeightedOnly = jobs.some(job => {
					const metrics = job.metrics || {};
					const hasW = metrics.recall_weighted !== undefined || metrics.test?.recall_weighted !== undefined;
					const hasU = metrics.recall !== undefined || metrics.test?.recall !== undefined;
					return hasW && !hasU;
				});
				if (!hasWeightedOnly) return false;
			}
			return true;
		});
	}, [problemType, jobs]);

	const sortedJobs = useMemo(() => {
		const sorted = [...jobs];
		sorted.sort((a, b) => {
			let aValue: any;
			let bValue: any;

			if (sortConfig.key === 'created_at') {
				aValue = a.created_at ? new Date(a.created_at).getTime() : 0;
				bValue = b.created_at ? new Date(b.created_at).getTime() : 0;
			} else if (sortConfig.key === 'model_type') {
				aValue = a.model_type || '';
				bValue = b.model_type || '';
			} else if (sortConfig.key === 'status') {
				aValue = a.status || '';
				bValue = b.status || '';
			} else {
				// Metric sort
				const aMetrics = a.metrics || {};
				const bMetrics = b.metrics || {};
				// Handle both simple and weighted keys for backward compatibility
				const getMetric = (m: Record<string, any>, k: string) => {
					if (m[k] !== undefined) return m[k];
					if (k === 'f1_weighted') return m['f1'];
					if (k === 'precision_weighted') return m['precision'];
					if (k === 'recall_weighted') return m['recall'];
					return undefined;
				};
				aValue = getMetric(aMetrics, sortConfig.key) ?? -Infinity;
				bValue = getMetric(bMetrics, sortConfig.key) ?? -Infinity;
			}

			if (aValue < bValue) {
				return sortConfig.direction === 'asc' ? -1 : 1;
			}
			if (aValue > bValue) {
				return sortConfig.direction === 'asc' ? 1 : -1;
			}
			return 0;
		});
		return sorted;
	}, [jobs, sortConfig]);

	const handleSort = (key: string) => {
		setSortConfig((prev) => ({
			key,
			direction: prev.key === key && prev.direction === 'desc' ? 'asc' : 'desc',
		}));
	};

	const getSortIcon = (key: string) => {
		if (sortConfig.key !== key) return <span style={{ opacity: 0.3 }}>↕</span>;
		return sortConfig.direction === 'asc' ? '↑' : '↓';
	};

	if (jobs.length === 0) {
		return (
			<div style={{ padding: '2rem', textAlign: 'center', color: 'rgba(148, 163, 184, 0.8)' }}>
				No training jobs available for comparison.
			</div>
		);
	}

	return (
		<div className="canvas-table-container" style={{ overflowX: 'auto' }}>
			<table className="canvas-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
				<thead>
					<tr>
						<th
							onClick={() => handleSort('model_type')}
							style={{ cursor: 'pointer', textAlign: 'left', padding: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}
						>
							Model {getSortIcon('model_type')}
						</th>
						<th style={{ textAlign: 'left', padding: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}>
							Target
						</th>
						<th
							onClick={() => handleSort('status')}
							style={{ cursor: 'pointer', textAlign: 'left', padding: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}
						>
							Status {getSortIcon('status')}
						</th>
						{metricColumns.map((col) => (
							<th
								key={col.key}
								onClick={() => handleSort(col.key)}
								style={{ cursor: 'pointer', textAlign: 'right', padding: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}
							>
								{col.label} {getSortIcon(col.key)}
							</th>
						))}
						<th
							onClick={() => handleSort('created_at')}
							style={{ cursor: 'pointer', textAlign: 'right', padding: '0.75rem', color: '#e2e8f0', fontWeight: 600 }}
						>
							Created {getSortIcon('created_at')}
						</th>
					</tr>
				</thead>
				<tbody>
					{sortedJobs.map((job) => {
						const metrics = job.metrics || {};
						const isSuccess = job.status === 'succeeded';
						
						// Helper to find metric value in nested structure (test > validation > train)
						const findMetricValue = (key: string) => {
							// 1. Try direct access (legacy)
							if (metrics[key] !== undefined) return metrics[key];
							
							// 2. Try nested buckets (test -> validation -> train)
							const buckets = [metrics.test, metrics.validation, metrics.train];
							for (const bucket of buckets) {
								if (bucket && typeof bucket === 'object' && bucket[key] !== undefined) {
									return bucket[key];
								}
							}
							
							// 3. Try cross-validation mean
							if (metrics.cross_validation?.metrics?.mean?.[key] !== undefined) {
								return metrics.cross_validation.metrics.mean[key];
							}
							
							return undefined;
						};

						return (
							<tr key={job.id} style={{ borderTop: '1px solid rgba(148, 163, 184, 0.1)' }}>
								<td style={{ padding: '0.75rem' }}>
									<div style={{ fontWeight: 500, color: '#e2e8f0' }}>
										{job.model_type.replace(/_/g, ' ')}
									</div>
									<div style={{ fontSize: '0.75rem', color: 'rgba(148, 163, 184, 0.7)' }}>
										ID: {job.id.slice(0, 8)}
									</div>
								</td>
								<td style={{ padding: '0.75rem', color: '#e2e8f0' }}>
									{job.metadata?.target_column || '—'}
								</td>
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
										{job.status}
									</span>
								</td>
								{metricColumns.map((col) => {
									// Try exact key, then fallback for backward compatibility
									let val = findMetricValue(col.key);
									if (val === undefined) {
										if (col.key === 'f1_weighted') val = findMetricValue('f1');
										else if (col.key === 'precision_weighted') val = findMetricValue('precision');
										else if (col.key === 'recall_weighted') val = findMetricValue('recall');
										else if (col.key === 'roc_auc') val = findMetricValue('roc_auc_weighted');
									}
									
									const hasVal = val !== undefined && val !== null;
									return (
										<td key={col.key} style={{ padding: '0.75rem', textAlign: 'right' }}>
											{isSuccess && hasVal ? (
												<span style={{ fontFamily: 'monospace', color: '#e2e8f0' }}>
													{formatMetricValue(val)}
												</span>
											) : (
												<span style={{ color: 'rgba(148, 163, 184, 0.3)' }}>—</span>
											)}
										</td>
									);
								})}
								<td style={{ padding: '0.75rem', textAlign: 'right', color: 'rgba(148, 163, 184, 0.8)' }}>
									{job.created_at ? formatRelativeTime(job.created_at) : '—'}
								</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
};
