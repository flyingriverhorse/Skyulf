import {
  useQuery,
  useMutation,
  useInfiniteQuery,
  useQueryClient,
  type QueryClient,
} from '@tanstack/react-query';
import { apiClient } from '../api/client';

/**
 * React Query hooks for the Model Registry (`/api/registry/*`) and the
 * single deployment endpoint used from the Registry page
 * (`POST /api/deployment/deploy/{jobId}`).
 *
 * Centralising these here keeps cache invalidation in one place: every
 * mutation that mutates a model row is responsible for invalidating the
 * `models` list so all consumers re-render with fresh data.
 */

export interface ArtifactResponse {
  storage_type: string;
  base_uri: string;
  files: string[];
}

export interface ModelVersion {
  job_id: string;
  pipeline_id: string;
  node_id: string;
  model_type: string;
  version: number | string;
  source: string;
  status: string;
  metrics: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
  created_at: string;
  artifact_uri: string;
  is_deployed: boolean;
  deployment_id?: number;
}

export interface ModelRegistryEntry {
  model_type: string;
  dataset_id: string;
  dataset_name: string;
  dataset_type?: string;
  latest_version: ModelVersion | null;
  versions: ModelVersion[];
  deployment_count: number;
}

export interface RegistryStats {
  total_models: number;
  total_versions: number;
  active_deployments: number;
}

export const registryKeys = {
  all: ['registry'] as const,
  stats: () => [...registryKeys.all, 'stats'] as const,
  models: () => [...registryKeys.all, 'models'] as const,
  artifacts: (jobId: string) => [...registryKeys.all, 'artifacts', jobId] as const,
};

const STATS_STALE_MS = 30_000;
const MODELS_PAGE_SIZE = 10;

const invalidateRegistry = (qc: QueryClient) => {
  void qc.invalidateQueries({ queryKey: registryKeys.models() });
  void qc.invalidateQueries({ queryKey: registryKeys.stats() });
};

export const useRegistryStats = () =>
  useQuery({
    queryKey: registryKeys.stats(),
    queryFn: async () => {
      const res = await apiClient.get<RegistryStats>('/registry/stats');
      return res.data;
    },
    staleTime: STATS_STALE_MS,
  });

/**
 * Paginated model list. Pages are appended client-side; React Query keeps
 * each page in cache so deploy mutations can invalidate the whole list at
 * once and trigger a clean refetch from page 0.
 */
export const useRegistryModels = () =>
  useInfiniteQuery({
    queryKey: registryKeys.models(),
    initialPageParam: 0,
    queryFn: async ({ pageParam }) => {
      const skip = (pageParam as number) * MODELS_PAGE_SIZE;
      const params = new URLSearchParams({
        skip: String(skip),
        limit: String(MODELS_PAGE_SIZE),
      });
      const res = await apiClient.get<ModelRegistryEntry[]>(
        `/registry/models?${params.toString()}`,
      );
      return res.data;
    },
    getNextPageParam: (lastPage, allPages) =>
      lastPage.length < MODELS_PAGE_SIZE ? undefined : allPages.length,
  });

/** Artifacts for a single training job; only fires when `jobId` is set. */
export const useArtifacts = (jobId: string | null) =>
  useQuery({
    queryKey: jobId ? registryKeys.artifacts(jobId) : ['registry', 'artifacts', 'idle'],
    enabled: !!jobId,
    queryFn: async () => {
      const res = await apiClient.get<ArtifactResponse>(`/registry/artifacts/${jobId}`);
      return res.data;
    },
  });

export const useDeployModel = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string) => {
      const res = await apiClient.post(`/deployment/deploy/${jobId}`);
      return res.data;
    },
    onSuccess: () => invalidateRegistry(qc),
  });
};
