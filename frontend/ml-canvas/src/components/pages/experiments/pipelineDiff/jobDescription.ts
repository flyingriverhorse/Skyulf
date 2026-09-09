import { shortRunId } from '../../ExperimentsPage/utils/jobMeta';
import type { JobLite } from '../pipelineDiffLayout';

export const formatTimestamp = (timestamp?: string): string => {
  if (!timestamp) return 'Unknown time';
  const value = new Date(timestamp);
  return Number.isNaN(value.getTime()) ? timestamp : value.toLocaleString();
};

export const describeJob = (job: JobLite): string => {
  const dataset = job.dataset_name ?? 'Unknown dataset';
  const model = job.model_type ?? 'Unknown model';
  return `${dataset} · ${model} · ${formatTimestamp(job.created_at)}`;
};

export const snapshotMessage = (role: 'Baseline' | 'Candidate', job: JobLite, detail: string): string =>
  `${role} run ${shortRunId(job)} (${describeJob(job)}) ${detail}. Re-run the pipeline or save the canvas snapshot, then compare again.`;
