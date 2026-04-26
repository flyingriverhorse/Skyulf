import { useQuery, useMutation, useQueryClient, type Query } from '@tanstack/react-query';
import { DatasetService } from '../api/datasets';
import type { Dataset, DataSourceCreate, IngestionJobResponse } from '../types/api';

/**
 * React Query hooks for the dataset surface (`/data/api/sources` +
 * `/api/ingestion/*`). Centralised so:
 *   1. Multiple Dataset nodes / pages don't each re-fetch the list.
 *   2. Mutations invalidate the same cache keys consistently.
 *   3. Polling for in-flight ingestion jobs is encoded once.
 *
 * Keys are exported so feature code can invalidate explicitly when
 * the standard mutation hooks here are not used (e.g. file upload
 * inside `FileUpload.tsx`).
 */
export const datasetKeys = {
  all: ['datasets'] as const,
  lists: () => [...datasetKeys.all, 'list'] as const,
  list: (variant: 'all' | 'usable') => [...datasetKeys.lists(), variant] as const,
  details: () => [...datasetKeys.all, 'detail'] as const,
  detail: (id: string) => [...datasetKeys.details(), id] as const,
};

const STALE_LIST_MS = 30_000; // datasets list rarely changes mid-session

/** All datasets (incl. failed / pending). Used by DataSources page. */
export const useDatasets = (options?: {
  refetchInterval?: number | false | ((query: Query<Dataset[]>) => number | false);
}) =>
  useQuery({
    queryKey: datasetKeys.list('all'),
    queryFn: () => DatasetService.getAll(),
    staleTime: STALE_LIST_MS,
    ...(options?.refetchInterval !== undefined ? { refetchInterval: options.refetchInterval } : {}),
  });

/** Only successfully-ingested datasets. Used by canvas Dataset nodes + EDA. */
export const useUsableDatasets = () =>
  useQuery({
    queryKey: datasetKeys.list('usable'),
    queryFn: () => DatasetService.getUsable(),
    staleTime: STALE_LIST_MS,
  });

/** Single dataset by id. */
export const useDataset = (id: string | undefined) =>
  useQuery({
    queryKey: datasetKeys.detail(id ?? ''),
    queryFn: () => DatasetService.getById(id!),
    enabled: !!id,
    staleTime: STALE_LIST_MS,
  });

/**
 * Whether the current `useDatasets` payload contains any in-flight
 * ingestion jobs — used to decide whether to keep polling.
 */
export const hasPendingIngestion = (datasets: Dataset[] | undefined): boolean =>
  (datasets ?? []).some((d) => {
    const status = d.source_metadata?.ingestion_status?.status;
    return status === 'pending' || status === 'processing';
  });

// ── Mutations ──────────────────────────────────────────────────────

function invalidateLists(qc: ReturnType<typeof useQueryClient>) {
  void qc.invalidateQueries({ queryKey: datasetKeys.lists() });
}

export const useDeleteDataset = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => DatasetService.delete(id),
    onSuccess: () => {
      invalidateLists(qc);
    },
  });
};

export const useCancelIngestion = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => DatasetService.cancelIngestion(id),
    onSuccess: () => {
      invalidateLists(qc);
    },
  });
};

export const useCreateDataSource = () => {
  const qc = useQueryClient();
  return useMutation<IngestionJobResponse, Error, DataSourceCreate>({
    mutationFn: (payload) => DatasetService.createSource(payload),
    onSuccess: () => {
      invalidateLists(qc);
    },
  });
};

export const useUploadDataset = () => {
  const qc = useQueryClient();
  return useMutation<IngestionJobResponse, Error, File>({
    mutationFn: (file) => DatasetService.upload(file),
    onSuccess: () => {
      invalidateLists(qc);
    },
  });
};
