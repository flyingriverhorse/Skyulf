import type { JobInfo } from '../../../core/api/jobs';
import type { RegistryItem } from '../../../core/api/registry';
import type { SubmittedRun } from '../../../core/types/runFeedback';
import type { TaskType } from '../../../core/types/taskType';
import { getEnsembleSubTask } from '../../../core/utils/format';
import { getTaskForModelType } from '../../pages/ExperimentsPage/utils/jobMeta';

const TASK_LABELS: Record<TaskType, string> = {
  classification: 'classification',
  regression: 'regression',
  text_classification: 'text classification',
  segmentation: 'segmentation',
  ensemble: 'ensemble',
};

export type EnsembleSubFilter = 'all' | 'classification' | 'regression';

export interface HistoryFilterValues {
  searchQuery: string;
  statusFilter: string;
  modelFilter: string;
}

interface HistorySelection {
  jobs: JobInfo[];
  runJobs: Record<string, JobInfo>;
  inspectedRun: SubmittedRun | null;
  activeTab: TaskType;
  ensembleSubFilter: EnsembleSubFilter;
  registryItems: RegistryItem[];
}

/** Submitted snapshots take precedence over stale history records. */
function findRunJob(id: string, runJobs: Record<string, JobInfo>, jobs: JobInfo[]) {
  return runJobs[id] ?? jobs.find(job => job.job_id === id);
}

/** Inspection follows receipt order; ordinary history follows task and ensemble scope. */
export function selectHistoryJobs({ jobs, runJobs, inspectedRun, activeTab,
  ensembleSubFilter, registryItems }: HistorySelection): JobInfo[] {
  if (inspectedRun) {
    return inspectedRun.jobIds.map(id => findRunJob(id, runJobs, jobs))
      .filter((job): job is JobInfo => Boolean(job));
  }
  return jobs
    .filter(job => getTaskForModelType(job.model_type, registryItems) === activeTab)
    .filter(job => activeTab !== 'ensemble' || ensembleSubFilter === 'all'
      || getEnsembleSubTask(job.model_type) === ensembleSubFilter);
}

/** Facets reflect the selected task before search filtering, in first-seen order. */
export function getHistoryFacets(tabJobs: JobInfo[]) {
  const modelTypes = [...new Set(tabJobs.map(job => job.model_type).filter(Boolean))] as string[];
  const statuses = [...new Set(tabJobs.map(job => job.status))];
  return { modelTypes, statuses };
}

/** Search preserves the dataset-name fallback and literal, case-insensitive matching. */
function matchesSearch(job: JobInfo, query: string): boolean {
  const q = query.toLowerCase();
  const matchesId = job.job_id.toLowerCase().includes(q);
  const matchesDataset = (job.dataset_name || job.dataset_id || '').toLowerCase().includes(q);
  const matchesModel = (job.model_type || '').toLowerCase().includes(q);
  return matchesId || matchesDataset || matchesModel;
}

/** Inspected runs bypass all retained history filters. */
export function filterHistoryJobs(jobs: JobInfo[], filters: HistoryFilterValues, inspectedRun: SubmittedRun | null) {
  if (inspectedRun) return jobs;
  const { statusFilter, modelFilter, searchQuery } = filters;
  return jobs.filter(job => {
    if (statusFilter !== 'all' && job.status !== statusFilter) return false;
    if (modelFilter !== 'all' && job.model_type !== modelFilter) return false;
    if (searchQuery && !matchesSearch(job, searchQuery)) return false;
    return true;
  });
}

export interface RunProgress {
  total: number;
  doneCount: number;
  pct: number;
  isDone: boolean;
}

/** Missing branches stay unfinished; progress covers the full overlapping active run. */
export function getRunProgress(activeParallelRun: { jobIds: string[] } | null,
  inspectedRun: SubmittedRun | null, runJobs: Record<string, JobInfo>, jobs: JobInfo[]): RunProgress | null {
  if (!activeParallelRun) return null;
  if (inspectedRun && !inspectedRun.jobIds.some(id => activeParallelRun.jobIds.includes(id))) return null;
  const total = activeParallelRun.jobIds.length;
  const terminal = new Set(['completed', 'succeeded', 'failed', 'cancelled']);
  const doneCount = activeParallelRun.jobIds.filter(id => {
    const job = findRunJob(id, runJobs, jobs);
    return job && terminal.has(job.status);
  }).length;
  return { total, doneCount, pct: Math.round((doneCount / total) * 100), isDone: doneCount === total };
}

/** Empty submitted receipts have a distinct recovery message from history filters. */
export function getEmptyHistoryMessage(inspectedRun: SubmittedRun | null,
  { searchQuery, statusFilter, modelFilter }: HistoryFilterValues, activeTab: TaskType): string {
  if (inspectedRun) return 'Waiting for the submitted jobs to appear. Use Refresh to check again.';
  if (searchQuery || statusFilter !== 'all' || modelFilter !== 'all') return 'No jobs match the current filters.';
  return `No ${TASK_LABELS[activeTab]} jobs found.`;
}
