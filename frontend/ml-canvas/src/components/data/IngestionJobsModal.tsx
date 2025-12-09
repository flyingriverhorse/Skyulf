import React from 'react';
import { X, Loader2, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Dataset } from '../../core/types/api';

interface IngestionJobsModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasets: Dataset[];
}

export const IngestionJobsModal: React.FC<IngestionJobsModalProps> = ({ isOpen, onClose, datasets }) => {
  if (!isOpen) return null;

  // Filter to show only items that have ingestion status or are recent
  // For now, we show all as "jobs"
  const jobs = datasets.map(d => ({
    id: d.id,
    name: d.name,
    status: d.source_metadata?.ingestion_status?.status || 'completed',
    created_at: d.created_at,
    message: d.source_metadata?.ingestion_status?.error || 'Ingestion completed successfully'
  })).sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl w-full max-w-3xl max-h-[80vh] flex flex-col border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in-95 duration-200">
        
        <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">Data Ingestion Jobs</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-500 dark:hover:text-slate-300">
            <X size={24} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {jobs.length === 0 ? (
            <div className="text-center text-slate-500 py-8">No ingestion jobs found.</div>
          ) : (
            <div className="space-y-4">
              {jobs.map((job) => (
                <div key={job.id} className="flex items-start gap-4 p-4 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50">
                  <div className="mt-1">
                    {job.status === 'processing' || job.status === 'pending' ? (
                      <Loader2 className="text-blue-500 animate-spin" size={20} />
                    ) : job.status === 'failed' ? (
                      <XCircle className="text-red-500" size={20} />
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
                    {job.status === 'failed' && (
                      <p className="text-sm text-red-600 dark:text-red-400 mt-1 bg-red-50 dark:bg-red-900/10 p-2 rounded">
                        {job.message}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
