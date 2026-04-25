import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useDatasetSchema } from './useDatasetSchema';
import * as client from '../api/client';

const wrapper = (qc: QueryClient): React.FC<{ children: React.ReactNode }> => {
  const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <QueryClientProvider client={qc}>{children}</QueryClientProvider>
  );
  Wrapper.displayName = 'TestQueryClientProvider';
  return Wrapper;
};

const mkQc = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false, gcTime: 0, staleTime: 0 },
    },
  });

describe('useDatasetSchema', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('is disabled (does not fetch) when sourceId is undefined', () => {
    const fetchSpy = vi.spyOn(client, 'fetchDatasetProfile');
    const { result } = renderHook(() => useDatasetSchema(undefined), {
      wrapper: wrapper(mkQc()),
    });
    expect(result.current.fetchStatus).toBe('idle');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it('fetches and returns the profile for a real sourceId', async () => {
    const profile = { columns: {}, row_count: 100 };
    const fetchSpy = vi
      .spyOn(client, 'fetchDatasetProfile')
      .mockResolvedValue(profile as never);

    const { result } = renderHook(() => useDatasetSchema('ds-1'), {
      wrapper: wrapper(mkQc()),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(fetchSpy).toHaveBeenCalledWith('ds-1');
    expect(result.current.data).toEqual(profile);
  });

  it('surfaces errors via isError', async () => {
    vi.spyOn(client, 'fetchDatasetProfile').mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useDatasetSchema('ds-bad'), {
      wrapper: wrapper(mkQc()),
    });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect((result.current.error as Error).message).toBe('boom');
  });
});
