import React from 'react';
import { Loader2, CheckCircle, XCircle, Clock, Ban } from 'lucide-react';
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
}

export const IngestionJobsModal: React.FC<IngestionJobsModalProps> = ({ isOpen, onClose, datasets, onRefresh }) => {
  const cancelMutation = useCancelIngestion();
  const cancellingId = cancelMutation.isPending ? cancelMutation.variables ?? null : null;
  const confirm = useConfirm();

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

  // Filter to show only items that have ingestion status or are recent
  // For now, we show all as "jobs"
  const jobs = datasets.map(d => ({
    id: d.id,
    name: d.name,
    status: d.source_metadata?.ingestion_status?.status || 'completed',
    created_at: d.created_at,
    message: d.source_metadata?.ingestion_status?.error ||
             (d.source_metadata?.ingestion_status?.status === 'cancelled' ? 'Ingestion cancelled' : 'Ingestion completed successfully')
  })).sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  return (
    <ModalShell isOpen={isOpen} onClose={onClose} title="Data Ingestion Jobs" size="3xl">
      <div className="p-6 max-h-[70vh] flex flex-col">
        {jobs.length === 0 ? (
          <div className="text-center text-slate-500 py-8">No ingestion jobs found.</div>
        ) : (
          <VirtualList
            items={jobs}
            getKey={(job) => job.id}
            estimateSize={108}
            className="flex-1 overflow-y-auto pr-1"
            renderItem={(job) => (
              <div className="pb-4">
                <div className="flex items-start gap-4 p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                <div className="mt-1">
                  {job.status === 'processing' || job.status === 'pending' ? (
                    <Loader2 className="text-blue-500 animate-spin" size={20} />
                  ) : job.status === 'failed' ? (
                    <XCircle className="text-red-500" size={20} />
                  ) : job.status === 'cancelled' ? (
                    <Ban className="text-slate-500" size={20} />
                  ) : (
                    <CheckCircle className="text-green-500" size={20} />
                  )}
                </div>
                <div className="flex-1">
                  <div className="flex justify-between items-start">
                    <h3 className="font-medium text-slate-900 dark:text-slate-100">{job.name}</h3>
                    <span className="text-xs text-slate-500 flex items-center gap-1">
                      <Clock size={12} />
                      {new Date(job.created_at).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                    Status: <span className="capitalize">{job.status}</span>
                  </p>
                  {(job.status === 'failed' || job.status === 'cancelled') && (
                    <p className={`text-sm mt-1 p-2 rounded ${
                      job.status === 'failed'
                        ? 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/10'
                        : 'text-slate-600 dark:text-slate-400 bg-slate-100 dark:bg-slate-800'
                    }`}>
                      {job.message}
                    </p>
                  )}
                </div>
                {(job.status === 'processing' || job.status === 'pending') && (
                  <button
                    onClick={() => { void handleCancel(job.id); }}
                    disabled={cancellingId === job.id}
                    className="p-2 text-slate-400 hover:text-orange-600 hover:bg-orange-50 dark:hover:bg-orange-900/20 rounded-md transition-colors disabled:opacity-50"
                    title="Cancel Ingestion"
                  >
                    {cancellingId === job.id ? (
                      <Loader2 className="animate-spin" size={20} />
                    ) : (
                      <Ban size={20} />
                    )}
                  </button>
                )}
                </div>
              </div>
            )}
          />
        )}
      </div>
    </ModalShell>
  );
};
