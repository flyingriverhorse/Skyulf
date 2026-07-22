import { useQuery } from '@tanstack/react-query';
import type { AxiosError } from 'axios';
import { fetchDatasetProfile } from '../api/client';

export const useDatasetSchema = (sourceId: string | undefined) => {
  return useQuery({
    queryKey: ['dataset-schema', sourceId],
    queryFn: () => fetchDatasetProfile(sourceId!),
    enabled: !!sourceId,
    staleTime: 1000 * 60 * 5, // 5 minutes
    // A failed dataset-schema fetch is only worth retrying if it's a
    // transient server-side issue (5xx). Client errors (404 = dataset no
    // longer exists, or any other 4xx) won't be fixed by retrying and
    // otherwise spam the backend with repeated failing requests per node.
    retry: (failureCount, error) => {
      const status = (error as AxiosError)?.response?.status;
      const isServerError = status !== undefined && status >= 500;
      return isServerError && failureCount < 3;
    },
  });
};
