import { useQuery } from '@tanstack/react-query';
import { fetchUploadConfig, type UploadConfig } from '../api/client';

/**
 * Server-side upload limits (max size + allowed extensions), fetched once
 * per session from `GET /api/config`. Callers keep hardcoded fallbacks for
 * the brief window before the first response (and for offline/dev use), so
 * this hook's absence of data must never block the upload UI.
 */
export const useUploadConfig = () =>
  useQuery<UploadConfig>({
    queryKey: ['upload-config'],
    queryFn: fetchUploadConfig,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
