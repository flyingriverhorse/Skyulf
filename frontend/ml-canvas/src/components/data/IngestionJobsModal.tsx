import React from 'react';
import { Loader2, CheckCircle, XCircle, Clock, Ban, RefreshCw } from 'lucide-react';
import { Dataset } from '../../core/types/api';
import { useCancelIngestion } from '../../core/hooks/useDatasets';
import { ModalShell, useConfirm } from '../shared';
import { VirtualList } from '../shared/VirtualList';
import { toast } from '../../core/toast';

interface IngestionJobsModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasets: Dataset[];
  onRefresh?: () => void;
  onRetry?: (dataset: Dataset) => Promise<void> | void;
}

interface IngestionRow {
  dataset: Dataset;
  id: string;
  name: string;
  status: string;
  createdAt: string;
  message: string;
}

const ACTIVE_STATUSES = new Set(['pending', 'processing']);

const toLifecycleMessage = (dataset: Dataset): string => {
  const ingestionStatus = dataset.source_metadata?.ingestion_status;
  const status = ingestionStatus?.status || 'completed';

  if (status === 'failed') {
    return ingestionStatus?.message || ingestionStatus?.error || 'Ingestion failed';
  }

  if (status === 'cancelled') {
    return ingestionStatus?.message || ingestionStatus?.error || 'Ingestion cancelled';
  }

  if (status === 'pending') {
    return 'Queued for ingestion';
  }

  if (status === 'processing') {
    return 'Processing ingestion';
  }

  return ingestionStatus?.message || 'Ingestion completed successfully';
};

/** Shows ingestion activity split into active work and terminal history. */
export const IngestionJobsModal: React.FC<IngestionJobsModalProps> = ({ isOpen, onClose, datasets, onRefresh, onRetry }) => {
  const cancelMutation = useCancelIngestion();
  const cancellingId = cancelMutation.isPending ? cancelMutation.variables ?? null : null;
  const confirm = useConfirm();
  const [retryingId, setRetryingId] = React.useState<string | null>(null);

  const handleCancel = async (id: string) => {
    const ok = await confirm({
      title: 'Cancel ingestion job?',
      message: 'Are you sure you want to cancel this ingestion job?',
      confirmLabel: 'Cancel job',
      variant: 'danger',
    });
    if (!ok) return;

    try {
      await cancelMutation.mutateAsync(id);
      // Mutation invalidates the dataset list cache; the optional callback
      // is kept for callers that need a manual refresh signal.
      if (onRefresh) onRefresh();
    } catch (error) {
      console.error('Failed to cancel ingestion:', error);
      toast.error('Failed to cancel ingestion');
    }
  };

  const handleRetry = async (dataset: Dataset) => {
    if (!onRetry) return;

    setRetryingId(dataset.id);
    try {
      await onRetry(dataset);
    } finally {
      setRetryingId(null);
    }
  };

  const jobs: IngestionRow[] = datasets
    .map((dataset) => {
      const status = dataset.source_metadata?.ingestion_status?.status || 'completed';
      return {
        dataset,
        id: dataset.id,
        name: dataset.name,
        status,
        createdAt: dataset.created_at,
        message: toLifecycleMessage(dataset),
      };
    })
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

  const activeJobs = jobs.filter((job) => ACTIVE_STATUSES.has(job.status));
  const historyJobs = jobs.filter((job) => !ACTIVE_STATUSES.has(job.status));

  const renderJob = (job: IngestionRow) => {
    const isActive = ACTIVE_STATUSES.has(job.status);
    const isFailed = job.status === 'failed';
    const isCancelled = job.status === 'cancelled';
    const showCancel = isActive;
    const showRetry = isFailed && onRetry;

    return (
      <div className="pb-4">
        <div className="flex items-start gap-4 p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
          <div className="mt-1">
            {job.status === 'processing' || job.status === 'pending' ? (
              <Loader2 className="text-blue-500 animate-spin" size={20} aria-hidden="true" />
            ) : isFailed ? (
              <XCircle className="text-red-500" size={20} aria-hidden="true" />
            ) : isCancelled ? (
              <Ban className="text-slate-500" size={20} aria-hidden="true" />
            ) : (
              <CheckCircle className="text-green-500" size={20} aria-hidden="true" />
            )}
          </div>
          <div className="flex-1">
            <div className="flex justify-between items-start">
              <h4 className="font-medium text-slate-900 dark:text-slate-100">{job.name}</h4>
              <span className="text-xs text-slate-500 flex items-center gap-1">
                <Clock size={12} aria-hidden="true" />
                {new Date(job.createdAt).toLocaleString()}
              </span>
            </div>
            {isActive ? (
              <p role="status" aria-atomic="true" className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                {job.status === 'processing' ? 'Processing ingestion' : 'Queued for ingestion'}
              </p>
            ) : (
              <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                Status: <span className="capitalize">{job.status}</span>
              </p>
            )}
            {(isFailed || isCancelled) && (
              <p
                className={`text-sm mt-1 p-2 rounded ${
                  isFailed
                    ? 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/10'
                    : 'text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800'
                }`}
                role={isFailed ? 'alert' : 'status'}
                aria-atomic="true"
              >
                {job.message}
              </p>
            )}
          </div>
          {showCancel && (
            <button
              onClick={() => { void handleCancel(job.id); }}
              disabled={cancellingId === job.id}
              className="p-2 text-slate-400 hover:text-orange-600 hover:bg-orange-50 dark:hover:bg-orange-900/20 rounded-md transition-colors disabled:opacity-50"
              title="Cancel ingestion"
              aria-label="Cancel ingestion"
            >
              {cancellingId === job.id ? (
                <Loader2 className="animate-spin" size={20} aria-hidden="true" />
              ) : (
                <Ban size={20} aria-hidden="true" />
              )}
            </button>
          )}
          {showRetry && (
            <button
              onClick={() => { void handleRetry(job.dataset); }}
              disabled={retryingId === job.id}
              className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium text-blue-700 dark:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors disabled:opacity-50"
            >
              {retryingId === job.id ? (
                <Loader2 className="animate-spin" size={16} aria-hidden="true" />
              ) : (
                <RefreshCw size={16} aria-hidden="true" />
              )}
              Retry
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <ModalShell isOpen={isOpen} onClose={onClose} title="Data Ingestion Activity" size="3xl">
      <div className="p-6 max-h-[70vh] flex flex-col">
        {jobs.length === 0 ? (
          <div className="text-center text-slate-500 py-8">No ingestion activity found.</div>
        ) : (
          <div className="flex flex-col gap-6 overflow-y-auto pr-1">
            <section aria-label="Active ingestions">
              <div className="mb-3">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Active ingestions
                </h3>
              </div>
              {activeJobs.length === 0 ? (
                <p className="text-sm text-slate-500 dark:text-slate-400">No active ingestions.</p>
              ) : (
                <VirtualList
                  items={activeJobs}
                  getKey={(job) => job.id}
                  estimateSize={108}
                  className="space-y-4"
                  renderItem={renderJob}
                />
              )}
            </section>

            <section aria-label="Ingestion history">
              <div className="mb-3">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Ingestion history
                </h3>
              </div>
              {historyJobs.length === 0 ? (
                <p className="text-sm text-slate-500 dark:text-slate-400">No ingestion history yet.</p>
              ) : (
                <VirtualList
                  items={historyJobs}
                  getKey={(job) => job.id}
                  estimateSize={108}
                  className="space-y-4"
                  renderItem={renderJob}
                />
              )}
            </section>
          </div>
        )}
      </div>
    </ModalShell>
  );
};
