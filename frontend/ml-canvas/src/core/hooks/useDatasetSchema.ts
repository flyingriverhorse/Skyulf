import { useQuery } from '@tanstack/react-query';
import { fetchDatasetProfile } from '../api/client';

export const useDatasetSchema = (sourceId: string | undefined) => {
  return useQuery({
    queryKey: ['dataset-schema', sourceId],
    queryFn: () => fetchDatasetProfile(sourceId!),
    enabled: !!sourceId,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
};
