import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';

import { ConfirmProvider } from '../components/shared';
import { serializeOperationalContext } from '../core/utils/operationalContext';
import { ModelRegistry } from './ModelRegistry';
import {
  useArtifacts,
  useDeployModel,
  useRegistryModels,
  useRegistryStats,
  type ModelRegistryEntry,
  type ModelVersion,
} from '../core/hooks/useModelRegistry';

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

const makeVersion = (overrides: Partial<ModelVersion> & Pick<ModelVersion, 'job_id' | 'version'>): ModelVersion => ({
  pipeline_id: 'pipe-1',
  node_id: 'node-1',
  model_type: 'random_forest',
  source: 'training',
  status: 'completed',
  metrics: { accuracy: 0.9 },
  hyperparameters: {},
  created_at: '2026-01-01T00:00:00Z',
  artifact_uri: 's3://bucket/model.pkl',
  is_deployed: false,
  ...overrides,
});

const makeEntry = (overrides: Partial<ModelRegistryEntry> & { versions: ModelVersion[] }): ModelRegistryEntry => ({
  model_type: 'random_forest',
  dataset_id: 'ds-1',
  dataset_name: 'Iris Dataset',
  latest_version: overrides.versions[0] ?? null,
  deployment_count: overrides.versions.filter((v) => v.is_deployed).length,
  ...overrides,
});

function mockRegistryHooks({
  models,
  deployMutateAsync = vi.fn(),
  isDeployPending = false,
  deployVariables,
}: {
  models: ModelRegistryEntry[];
  deployMutateAsync?: ReturnType<typeof vi.fn>;
  isDeployPending?: boolean;
  deployVariables?: string;
}) {
  vi.mocked(useRegistryStats).mockReturnValue({
    data: { total_models: models.length, total_versions: models.reduce((n, m) => n + m.versions.length, 0), active_deployments: 1 },
    error: null,
    isFetching: false,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useRegistryModels).mockReturnValue({
    data: { pages: [models] },
    error: null,
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetching: false,
    isFetchingNextPage: false,
    refetch: vi.fn(),
  } as never);
  vi.mocked(useArtifacts).mockReturnValue({
    data: null,
    isFetching: false,
  } as never);
  vi.mocked(useDeployModel).mockReturnValue({
    isPending: isDeployPending,
    mutateAsync: deployMutateAsync,
    variables: deployVariables,
  } as never);
}

function renderRegistry(initialEntry = '/registry') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={client}>
        <ConfirmProvider>
          <ModelRegistry />
        </ConfirmProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

// jsdom does not implement IntersectionObserver; ModelRegistry's infinite-scroll
// sentinel needs a stub so mounting the component doesn't throw in tests.
class StubIntersectionObserver implements IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = '';
  readonly thresholds: ReadonlyArray<number> = [];
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
  takeRecords(): IntersectionObserverEntry[] { return []; }
}
vi.stubGlobal('IntersectionObserver', StubIntersectionObserver);

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
      <MemoryRouter>
        <QueryClientProvider client={client}>
          <ConfirmProvider>
            <ModelRegistry />
          </ConfirmProvider>
        </QueryClientProvider>
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: /retry/i }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });
});

describe('ModelRegistry lineage (OPS-002)', () => {
  it('does not render any client-only manual-deployment tracking control', () => {
    const version = makeVersion({ job_id: 'job-1', version: 1, is_deployed: true, deployment_id: 7 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version] })] });
    renderRegistry();

    // The pre-OPS-002 UI tracked "manual deployments" in localStorage via a
    // checkbox; that was a real lineage-consistency bug and must be gone.
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  });

  it('links the deployed version to its deployment record via RecordLink', () => {
    const version = makeVersion({ job_id: 'job-1', version: 3, is_deployed: true, deployment_id: 42 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version] })] });
    renderRegistry();

    const link = screen.getByRole('link', { name: /deployment #?42|Deployment 42/i });
    expect(link).toBeInTheDocument();
    expect(link.getAttribute('href')?.split('?')[0]).toBe('/deployments');
  });

  it('renders an explicit "no target available" note when the dataset id is unknown', () => {
    const version = makeVersion({ job_id: 'job-1', version: 1 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version], dataset_id: 'unknown', dataset_name: 'Unknown dataset' })] });
    renderRegistry();

    const note = screen.getByTitle('No target available');
    expect(note).toBeInTheDocument();
    expect(note).toHaveTextContent(/Unknown dataset/i);
  });

  it('opens the version dialog and lets the operator deploy a version, naming it exactly', async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockResolvedValue({});
    const version = makeVersion({ job_id: 'job-9', version: 2 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version] })], deployMutateAsync: mutateAsync });
    renderRegistry();

    await user.click(screen.getByRole('button', { name: /view versions/i }));
    await user.click(screen.getByRole('button', { name: /deploy version 2 \(job job-9\)/i }));

    // Confirmation must name the exact version being deployed.
    expect(await screen.findByText(/deploy version 2 \(job job-9\)/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /^deploy$/i }));

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledWith('job-9'));
  });

  it('scopes deploy failure to the affected version and allows retry-in-place without double submit', async () => {
    const user = userEvent.setup();
    const mutateAsync = vi.fn().mockRejectedValueOnce(new Error('boom')).mockResolvedValueOnce({});
    const version = makeVersion({ job_id: 'job-9', version: 2 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version] })], deployMutateAsync: mutateAsync });
    renderRegistry();

    await user.click(screen.getByRole('button', { name: /view versions/i }));
    await user.click(screen.getByRole('button', { name: /deploy version 2 \(job job-9\)/i }));
    await user.click(await screen.findByRole('button', { name: /^deploy$/i }));

    // Failure names exactly which version/job it affected.
    expect(await screen.findByText(/failed to deploy version 2 \(job job-9\)/i)).toBeInTheDocument();

    const retryButton = screen.getByRole('button', { name: /retry/i });
    await user.click(retryButton);

    await waitFor(() => expect(mutateAsync).toHaveBeenCalledTimes(2));
    expect(screen.queryByText(/failed to deploy version 2/i)).not.toBeInTheDocument();
  });

  it('deep-links directly to a modelVersion context and auto-opens its dialog', async () => {
    const version = makeVersion({ job_id: 'job-deep', version: 5 });
    mockRegistryHooks({ models: [makeEntry({ versions: [version] })] });

    const query = serializeOperationalContext({ ref: { kind: 'modelVersion', jobId: 'job-deep', version: '5' } });
    renderRegistry(`/registry${query}`);

    expect(await screen.findByText(/version history/i)).toBeInTheDocument();
    expect(screen.getAllByText(/v5/).length).toBeGreaterThan(0);
  });
});
