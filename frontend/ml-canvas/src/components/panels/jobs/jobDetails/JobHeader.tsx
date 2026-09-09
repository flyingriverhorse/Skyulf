import { useState, useMemo } from 'react';
import { X, ArrowLeft, Square, RotateCw, Info } from 'lucide-react';
import { JobInfo } from '../../../../core/api/jobs';
import { useJobStore } from '../../../../core/store/useJobStore';
import { isTerminalStatus } from '../../../../core/hooks/useJobPolling';
import { isEnsembleModelType, getEnsembleStrategy, getEnsembleSubTask } from '../../../../core/utils/format';
import { useConfirm } from '../../../shared';
import { toast } from '../../../../core/toast';

/** Whether recovery (retry) is currently supported for a job, and why not when it isn't. */
interface RetryAvailability {
  available: boolean;
  reason?: string;
}

const RETRYABLE_JOB_TYPES: ReadonlySet<JobInfo['job_type']> = new Set(['training', 'tuning']);
const RETRYABLE_STATUSES: ReadonlySet<JobInfo['status']> = new Set(['failed', 'cancelled']);

/**
 * Determines whether the Retry action is available for a job and, when it
 * isn't, why — so the UI can always state availability instead of silently
 * hiding the control (OPS-001 acceptance: "unavailable retry is not implied").
 */
function getRetryAvailability(job: JobInfo): RetryAvailability {
  if (!RETRYABLE_JOB_TYPES.has(job.job_type)) {
    return { available: false, reason: `Retry isn't available for ${job.job_type} jobs.` };
  }
  if (RETRYABLE_STATUSES.has(job.status)) {
    return { available: true };
  }
  if (job.status === 'completed' || job.status === 'succeeded') {
    return { available: false, reason: 'Retry is not needed — this job completed successfully.' };
  }
  return { available: false, reason: 'Retry becomes available once this job fails or is cancelled.' };
}

function useJobActions(job: JobInfo, onBack: () => void) {
  const { cancelJob, retryJob } = useJobStore();
  const confirm = useConfirm();
  const [isCancelling, setIsCancelling] = useState(false);
  const [isRetrying, setIsRetrying] = useState(false);
  const handleCancel = async () => {
    const ok = await confirm({
      title: 'Stop job?',
      message: 'Are you sure you want to stop this job?',
      confirmLabel: 'Stop',
      variant: 'danger',
    });
    if (!ok) return;
    setIsCancelling(true);
    try {
      await cancelJob(job.job_id);
    } catch (e) {
      toast.error('Failed to cancel job');
    } finally {
      setIsCancelling(false);
    }
  };

  const handleRetry = async () => {
    if (isRetrying) return; // Guard against a double-click firing two retry submissions.
    const ok = await confirm({
      title: 'Retry job?',
      message: 'This resubmits the same pipeline configuration as a new job.',
      confirmLabel: 'Retry',
    });
    if (!ok) return;
    setIsRetrying(true);
    try {
      const newJobId = await retryJob(job.job_id);
      toast.success(`Retry submitted as job ${newJobId.slice(0, 8)}`);
      onBack();
    } catch (e) {
      toast.error('Failed to retry job');
    } finally {
      setIsRetrying(false);
    }
  };
  return { isCancelling, isRetrying, handleCancel, handleRetry };
}

function JobRetryAction({ job, actions }: { job: JobInfo; actions: ReturnType<typeof useJobActions> }) {
  const { isRetrying, handleRetry } = actions;
  const retryAvailability = useMemo(() => getRetryAvailability(job), [job]);
  return (
    <>
      {isTerminalStatus(job.status) && (
        retryAvailability.available ? (
          <button
            onClick={() => { void handleRetry(); }}
            disabled={isRetrying}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/40 rounded text-xs font-medium transition-colors border border-blue-200 dark:border-blue-800"
          >
            <RotateCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
            {isRetrying ? 'Retrying...' : 'Retry'}
          </button>
        ) : (
          <span
            className="flex items-center gap-1 px-2 py-1.5 text-xs text-gray-500 dark:text-gray-400"
            title={retryAvailability.reason}
          >
            <Info className="w-3 h-3" />
            Retry unavailable
          </span>
        )
      )}
    </>
  );
}

export function JobHeader({ job, onBack, onClose }: { job: JobInfo; onBack: () => void; onClose: () => void }) {
  const actions = useJobActions(job, onBack);
  const { isCancelling, handleCancel } = actions;
  return (
    <div className="p-4 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center bg-gray-50 dark:bg-gray-800/50">
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded text-gray-500">
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div>
          <h2 className="font-semibold text-gray-800 dark:text-gray-100 flex items-center gap-2">
            Job Details
            <span
              className="text-xs font-normal text-gray-500 font-mono bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded"
              title={job.job_id}
            >
              <span className="sr-only">Job ID </span>
              {job.job_id.slice(0, 8)}
              <span className="sr-only">{job.job_id.slice(8)}</span>
            </span>
            {isEnsembleModelType(job.model_type) && (
              <span className="text-xs font-normal px-1.5 py-0.5 rounded border bg-violet-50 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300 border-violet-200 dark:border-violet-800 whitespace-nowrap">
                {getEnsembleStrategy(job.model_type)} · {getEnsembleSubTask(job.model_type) === 'regression' ? 'Regression' : 'Classification'}
              </span>
            )}
          </h2>
        </div>
      </div>
      <div className="flex items-center gap-2">
        {(job.status === 'running' || job.status === 'queued') && (
          <button
            onClick={() => { void handleCancel(); }}
            disabled={isCancelling}
            className="flex items-center gap-1 px-3 py-1.5 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-900/40 rounded text-xs font-medium transition-colors border border-red-200 dark:border-red-800"
          >
            <Square className="w-3 h-3 fill-current" />
            {isCancelling ? 'Stopping...' : 'Stop Job'}
          </button>
        )}
        <JobRetryAction job={job} actions={actions} />
        <button onClick={onClose} className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
