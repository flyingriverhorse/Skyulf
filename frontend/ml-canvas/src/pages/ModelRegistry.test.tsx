import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ConfirmProvider } from '../components/shared';
import { ModelRegistry } from './ModelRegistry';
import { useArtifacts, useDeployModel, useRegistryModels, useRegistryStats } from '../core/hooks/useModelRegistry';

vi.mock('../core/hooks/useModelRegistry', async () => {
  const actual = await vi.importActual<typeof import('../core/hooks/useModelRegistry')>('../core/hooks/useModelRegistry');
  return {
    ...actual,
    useArtifacts: vi.fn(),
    useDeployModel: vi.fn(),
    useRegistryModels: vi.fn(),
    useRegistryStats: vi.fn(),
  };
});

describe('ModelRegistry error retry', () => {
  it('renders a retry button when the registry query errors', () => {
    const refetch = vi.fn();
    vi.mocked(useRegistryStats).mockReturnValue({
      data: null,
      error: null,
      isFetching: false,
      refetch: vi.fn(),
    } as never);
    vi.mocked(useRegistryModels).mockReturnValue({
      data: undefined,
      error: new Error('registry failed'),
      fetchNextPage: vi.fn(),
      hasNextPage: false,
      isFetching: false,
      refetch,
    } as never);
    vi.mocked(useArtifacts).mockReturnValue({
      data: null,
      isFetching: false,
    } as never);
    vi.mocked(useDeployModel).mockReturnValue({
      isPending: false,
      mutateAsync: vi.fn(),
      variables: undefined,
    } as never);

    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={client}>
        <ConfirmProvider>
          <ModelRegistry />
        </ConfirmProvider>
      </QueryClientProvider>,
    );

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });
});
