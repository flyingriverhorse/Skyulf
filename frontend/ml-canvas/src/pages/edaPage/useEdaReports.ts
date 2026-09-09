import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { DatasetService } from '../../core/api/datasets';
import { EDAService } from '../../core/api/eda';
import { edaKeys } from '../../core/hooks/useEdaJobs';
import { buildEdaDatasetOptions } from '../../core/utils/edaDatasetOptions';

/** Keep report/history query keys, missing-report handling and pending polling together. */
export function useEdaReports(selectedDataset: number | null) {
  const datasetsQuery = useQuery({
    queryKey: edaKeys.datasets,
    queryFn: () => DatasetService.getUsable(),
  });
  // Memoize the fallback so dependent effects don't refire on every render.
  const datasets = useMemo(() => datasetsQuery.data ?? [], [datasetsQuery.data]);
  const datasetOptions = useMemo(() => buildEdaDatasetOptions(datasets), [datasets]);

  const reportQuery = useQuery({
    queryKey: edaKeys.report(selectedDataset ?? null),
    queryFn: async () => {
      try {
        return await EDAService.getLatestReport(selectedDataset!);
      } catch (err: unknown) {
        // 404 simply means "no report yet" — surface as null rather than a hard error.
        const status = (err as { response?: { status?: number; }; })?.response?.status;
        if (status === 404) return null;
        throw err;
      }
    },
    enabled: selectedDataset != null,
    // Auto-poll every 3 s while the backend job is PENDING; stop once it completes/fails.
    refetchInterval: (query) => {
      const data = query.state.data;
      return data && data.status === 'PENDING' ? 3000 : false;
    },
  });
  const report = reportQuery.data ?? null;
  const loading = reportQuery.isLoading;
  const error = reportQuery.isError ? 'Failed to load report' : null;

  const historyQuery = useQuery({
    queryKey: edaKeys.history(selectedDataset ?? null),
    queryFn: () => EDAService.getHistory(selectedDataset!),
    enabled: selectedDataset != null,
    // Refresh history alongside the report while a job is in flight.
    refetchInterval: report?.status === 'PENDING' ? 3000 : false,
  });
  const history = historyQuery.data ?? [];

  return { datasetsQuery, datasets, datasetOptions, reportQuery, report, loading, error, history };
}
