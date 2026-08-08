import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import type { Dataset } from '../core/types/api';
import {
  ANONYMOUS_ACTOR,
  pipelineVersionsApi,
  type AuditLogEntry,
  type AuditLogResponse,
} from '../core/api/pipelineVersions';
import { useUsableDatasets } from '../core/hooks/useDatasets';
import { AuditLogPage } from './AuditLogPage';

vi.mock('../core/hooks/useDatasets', async () => {
  const actual = await vi.importActual<typeof import('../core/hooks/useDatasets')>(
    '../core/hooks/useDatasets',
  );
  return {
    ...actual,
    useUsableDatasets: vi.fn(),
  };
});

vi.mock('../core/api/pipelineVersions', async () => {
  const actual = await vi.importActual<typeof import('../core/api/pipelineVersions')>(
    '../core/api/pipelineVersions',
  );
  return {
    ...actual,
    pipelineVersionsApi: {
      ...actual.pipelineVersionsApi,
      audit: vi.fn(),
    },
  };
});

const datasets: Dataset[] = [
  {
    id: 'dataset-alpha',
    name: 'Dataset Alpha',
    type: 'file',
    created_at: '2026-08-07T08:00:00.000Z',
  },
];

const auditEntries: AuditLogEntry[] = [
  {
    id: 3,
    version_int: 3,
    name: 'Latest auto save',
    note: null,
    kind: 'auto',
    user_id: 7,
    created_at: '2026-08-07T10:00:00.000Z',
    node_count: 12,
    edge_count: 8,
    diff: { nodes_added: ['node-c'], nodes_removed: [], nodes_modified: [], delta_node_count: 1 },
  },
  {
    id: 2,
    version_int: 2,
    name: 'Manual midpoint save',
    note: null,
    kind: 'manual',
    user_id: 9,
    created_at: '2026-08-05T10:00:00.000Z',
    node_count: 11,
    edge_count: 8,
    diff: { nodes_added: ['node-b'], nodes_removed: [], nodes_modified: [], delta_node_count: 1 },
  },
  {
    id: 1,
    version_int: 1,
    name: 'Anonymous bootstrap save',
    note: null,
    kind: 'manual',
    user_id: null,
    created_at: '2026-08-01T10:00:00.000Z',
    node_count: 10,
    edge_count: 8,
    diff: { nodes_added: ['node-a'], nodes_removed: [], nodes_modified: [], delta_node_count: 1 },
  },
];

const FACETS = {
  actors: ['7', '9'],
  kinds: ['auto', 'manual'],
  has_anonymous_actor: true,
} as const;

/** Build a response whose facets stay complete regardless of the filter. */
function response(entries: AuditLogEntry[], totalUnfiltered = 3): AuditLogResponse {
  return {
    dataset_source_id: 'dataset-alpha',
    total: entries.length,
    total_unfiltered: totalUnfiltered,
    facets: { ...FACETS, actors: [...FACETS.actors], kinds: [...FACETS.kinds] },
    filters: { actor: null, kind: null, created_after: null, created_before: null },
    entries,
  };
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <AuditLogPage />
    </QueryClientProvider>,
  );
}

describe('AuditLogPage server-side filtering', () => {
  beforeEach(() => {
    vi.mocked(useUsableDatasets).mockReturnValue({
      data: datasets,
      isLoading: false,
    } as never);
    vi.mocked(pipelineVersionsApi.audit).mockReset();
  });

  it('sends the action kind to the server rather than filtering the loaded page', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    audit.mockResolvedValue(response(auditEntries.slice(1)));
    fireEvent.change(screen.getByLabelText('Action kind'), { target: { value: 'manual' } });

    await waitFor(() => {
      expect(audit).toHaveBeenLastCalledWith(
        'dataset-alpha',
        50,
        expect.objectContaining({ kind: 'manual' }),
      );
    });
    await waitFor(() => {
      expect(screen.queryByText('Latest auto save')).not.toBeInTheDocument();
      expect(screen.getByText('Manual midpoint save')).toBeInTheDocument();
    });
  });

  it('sends the selected actor id to the server', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    audit.mockResolvedValue(response([auditEntries[1] as AuditLogEntry]));
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: '9' } });

    await waitFor(() => {
      expect(audit).toHaveBeenLastCalledWith(
        'dataset-alpha',
        50,
        expect.objectContaining({ actor: '9' }),
      );
    });
  });

  it('sends the anonymous sentinel for saves without a user id', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    audit.mockResolvedValue(response([auditEntries[2] as AuditLogEntry]));
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: ANONYMOUS_ACTOR } });

    await waitFor(() => {
      expect(audit).toHaveBeenLastCalledWith(
        'dataset-alpha',
        50,
        expect.objectContaining({ actor: ANONYMOUS_ACTOR }),
      );
    });
  });

  it('sends the time range as ISO bounds', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    audit.mockResolvedValue(response(auditEntries.slice(0, 2)));
    fireEvent.change(screen.getByLabelText('From time'), {
      target: { value: '2026-08-05T00:00' },
    });

    await waitFor(() => {
      expect(audit).toHaveBeenLastCalledWith(
        'dataset-alpha',
        50,
        expect.objectContaining({ createdAfter: '2026-08-05T00:00' }),
      );
    });
  });

  it('keeps every actor in the dropdown while a single actor is selected', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    audit.mockResolvedValue(response([auditEntries[1] as AuditLogEntry]));
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: '9' } });

    await waitFor(() => {
      expect(screen.queryByText('Latest auto save')).not.toBeInTheDocument();
    });
    // Facets come from the server's pre-filter pass, so user #7 must survive.
    expect(screen.getByRole('option', { name: 'user #7' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'user #9' })).toBeInTheDocument();
  });

  it('reports matching versus total counts once a filter is applied', async () => {
    const audit = vi.mocked(pipelineVersionsApi.audit);
    audit.mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');
    expect(
      screen.getByText(/Showing 3 of 3 saves for Dataset Alpha\./i),
    ).toBeInTheDocument();

    audit.mockResolvedValue(response([auditEntries[1] as AuditLogEntry]));
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: '9' } });

    await waitFor(() => {
      expect(
        screen.getByText(/Showing 1 of 1 matching saves for Dataset Alpha\. History total 3\./i),
      ).toBeInTheDocument();
    });
  });

  it('states that filters span the whole history, not just the page', async () => {
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response(auditEntries));

    renderPage();
    await screen.findByText('Latest auto save');

    expect(
      screen.getByText(/Filters are applied across the full history, not just this page\./i),
    ).toBeInTheDocument();
  });

  it('explains the dataset and window in the empty state', async () => {
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response([], 0));

    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/No saves recorded for Dataset Alpha yet\./i)).toBeInTheDocument();
    });
  });

  it('tells the user to widen filters when the history is not empty but nothing matches', async () => {
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response([], 3));

    renderPage();

    await waitFor(() => {
      expect(
        screen.getByText(/Filters were applied across all 3 saves/i),
      ).toBeInTheDocument();
    });
  });
});
