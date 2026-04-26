import { useMutation, useQueryClient } from '@tanstack/react-query';
import { EDAService } from '../api/eda';

/**
 * React Query keys for the EDA surface. Co-located here so any consumer
 * (page, modal, or sub-component) can invalidate the same cache entries
 * without depending on `EDAPage`.
 */
export const edaKeys = {
  datasets: ['eda', 'datasets'] as const,
  report: (id: number | null) => ['eda', 'report', id] as const,
  history: (id: number | null) => ['eda', 'history', id] as const,
  reportById: (id: number) => ['eda', 'reportById', id] as const,
};

/**
 * Cancel an EDA job and invalidate the per-dataset history + report
 * caches so the UI reflects the new status without a manual refresh.
 */
export const useCancelEdaJob = (datasetId: number | null) => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: number) => EDAService.cancelJob(jobId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: edaKeys.history(datasetId) });
      void qc.invalidateQueries({ queryKey: edaKeys.report(datasetId) });
    },
  });
};
