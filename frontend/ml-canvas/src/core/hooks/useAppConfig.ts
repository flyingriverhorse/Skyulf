import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';

interface AppConfig {
  demo_mode: boolean;
}

export const configKeys = {
  all: ['config'] as const,
};

/** Fetch public app configuration from the backend. */
export const useAppConfig = () =>
  useQuery<AppConfig>({
    queryKey: configKeys.all,
    queryFn: () => apiClient.get('/config').then((r) => r.data),
    staleTime: Infinity, // never changes during a session
  });
