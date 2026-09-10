import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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
import { toast } from '../core/toast';

vi.mock('../core/toast', () => ({ toast: { error: vi.fn() } }));

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

/** Let requests settle in a chosen order to pin the page's generation guard. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

describe('AuditLogPage server-side filtering', () => {
  beforeEach(() => {
    vi.mocked(useUsableDatasets).mockReturnValue({
      data: datasets,
      isLoading: false,
    } as never);
    vi.mocked(pipelineVersionsApi.audit).mockReset();
    vi.mocked(toast.error).mockClear();
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
    expect(screen.getByText(
      'Actor, action kind and time filters apply across the full history before the page limit.',
    )).toBeInTheDocument();
    expect(screen.queryByText(/Filters apply only to the loaded page;/)).not.toBeInTheDocument();
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

  it('selects the first numeric dataset id as a string and keeps the default limit', async () => {
    // Legacy numeric ids must match canvas saves without breaking picker labels.
    vi.mocked(useUsableDatasets).mockReturnValue({
      data: [{ ...datasets[0], id: 123 }], isLoading: false,
    } as never);
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response(auditEntries));
    renderPage();
    await screen.findByText('Latest auto save');
    expect(screen.getByLabelText('Dataset')).toHaveValue('123');
    expect(screen.getByRole('option', { name: 'Dataset Alpha (123)' })).toBeInTheDocument();
    expect(pipelineVersionsApi.audit).toHaveBeenCalledExactlyOnceWith('123', 50, {});
  });

  it('waits for datasets and disables refresh when the picker is empty', () => {
    // A missing dataset must never trigger a history request.
    vi.mocked(useUsableDatasets).mockReturnValue({ data: undefined, isLoading: true } as never);
    const page = renderPage();
    expect(screen.getByRole('option', { name: 'Loading…' })).toBeInTheDocument();
    expect(screen.getByLabelText('Dataset')).toBeDisabled();
    vi.mocked(useUsableDatasets).mockReturnValue({ data: [], isLoading: false } as never);
    page.rerender(<AuditLogPage />);
    expect(screen.getByRole('option', { name: 'No datasets' })).toBeInTheDocument();
    expect(screen.getByText('Pick a dataset to view its save history.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
    expect(pipelineVersionsApi.audit).not.toHaveBeenCalled();
  });

  it('preserves filters and local datetime strings across limit and dataset changes', async () => {
    // Changing the page window must not silently reset server-side filters.
    vi.mocked(useUsableDatasets).mockReturnValue({
      data: [...datasets, { ...datasets[0], id: 'dataset-beta', name: 'Dataset Beta' }],
      isLoading: false,
    } as never);
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response(auditEntries));
    renderPage();
    await screen.findByText('Latest auto save');
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: ANONYMOUS_ACTOR } });
    fireEvent.change(screen.getByLabelText('Action kind'), { target: { value: 'manual' } });
    fireEvent.change(screen.getByLabelText('From time'), { target: { value: '2026-08-01T12:30' } });
    fireEvent.change(screen.getByLabelText('To time'), { target: { value: '2026-08-07T19:45' } });
    fireEvent.click(screen.getByRole('button', { name: '200' }));
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: 'dataset-beta' } });
    await waitFor(() => expect(pipelineVersionsApi.audit).toHaveBeenLastCalledWith('dataset-beta', 200, {
      actor: ANONYMOUS_ACTOR, kind: 'manual', createdAfter: '2026-08-01T12:30', createdBefore: '2026-08-07T19:45',
    }));
    expect(screen.getByLabelText('Actor')).toHaveValue(ANONYMOUS_ACTOR);
    expect(screen.getByRole('option', { name: 'auto' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'user #7' })).toBeInTheDocument();
    expect(screen.getByText(/Page limit 200\./)).toBeInTheDocument();
  });

  it('omits cleared filters from the request', async () => {
    // All and empty inputs use omission, not literal sentinels or converted dates.
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response(auditEntries));
    renderPage();
    await screen.findByText('Latest auto save');
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: '7' } });
    fireEvent.change(screen.getByLabelText('Action kind'), { target: { value: 'auto' } });
    fireEvent.change(screen.getByLabelText('From time'), { target: { value: '2026-08-01T00:00' } });
    fireEvent.change(screen.getByLabelText('To time'), { target: { value: '2026-08-07T00:00' } });
    fireEvent.change(screen.getByLabelText('Actor'), { target: { value: 'all' } });
    fireEvent.change(screen.getByLabelText('Action kind'), { target: { value: 'all' } });
    fireEvent.change(screen.getByLabelText('From time'), { target: { value: '' } });
    fireEvent.change(screen.getByLabelText('To time'), { target: { value: '' } });
    await waitFor(() => expect(pipelineVersionsApi.audit).toHaveBeenLastCalledWith('dataset-alpha', 50, {}));
  });

  it('summarizes visible saves and renders row metadata and expandable node details', async () => {
    // Summaries count visible diffs while the oldest visible addition-only row is initial.
    const latest = { ...auditEntries[0]!, note: 'Important save', diff: {
      nodes_added: ['node-c', 'node-d'], nodes_removed: ['old-node'], nodes_modified: ['changed-node'], delta_node_count: 1,
    } };
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response([latest, ...auditEntries.slice(1)], 99));
    renderPage();
    await screen.findByText('Latest auto save');
    for (const [label, value] of [['Saves', '3'], ['Users', '3'], ['Nodes added', '4'], ['Nodes removed', '1'], ['Nodes modified', '1']]) {
      expect(within(screen.getByText(label!).parentElement!).getByText(value!)).toBeInTheDocument();
    }
    const newest = screen.getByRole('button', { name: /Latest auto save/ });
    const oldest = screen.getByRole('button', { name: /Anonymous bootstrap save/ });
    expect(within(newest).getByText(new Date(latest.created_at).toLocaleString())).toBeInTheDocument();
    expect(within(newest).getByText('“Important save”')).toBeInTheDocument();
    expect(within(newest).getByText('12 nodes / 8 edges')).toBeInTheDocument();
    expect(within(oldest).getByText('initial')).toBeInTheDocument();
    expect(within(oldest).queryByText('1 added')).not.toBeInTheDocument();
    fireEvent.click(newest);
    expect(screen.getByText('Added (2)')).toBeInTheDocument();
    expect(screen.getByText('old-node')).toBeInTheDocument();
    expect(screen.getByText('changed-node')).toBeInTheDocument();
    fireEvent.click(newest);
    expect(screen.queryByText('old-node')).not.toBeInTheDocument();
    fireEvent.click(oldest);
    expect(screen.getByText('node-a')).toBeInTheDocument();
  });

  it('does not mark the oldest visible row initial when it contains removals or modifications', async () => {
    // Genesis is determined from position plus diff, not version number.
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response([{ ...auditEntries[0]!,
      diff: { nodes_added: [], nodes_removed: ['old'], nodes_modified: ['changed'], delta_node_count: -1 },
    }]));
    renderPage();
    await screen.findByText('Latest auto save');
    expect(screen.queryByText('initial')).not.toBeInTheDocument();
    expect(screen.getByText('1 removed')).toBeInTheDocument();
    expect(screen.getByText('1 modified')).toBeInTheDocument();
  });

  it('renders unchanged non-genesis saves without expandable node content', async () => {
    // Empty diffs keep their explanatory label without fabricated detail lists.
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response([{ ...auditEntries[0]!,
      diff: { nodes_added: [], nodes_removed: [], nodes_modified: [], delta_node_count: 0 },
    }, auditEntries[2]!]));
    renderPage();
    await screen.findByText('Latest auto save');
    fireEvent.click(screen.getByRole('button', { name: /Latest auto save/ }));
    expect(screen.getByText('no node-level changes')).toBeInTheDocument();
    expect(screen.queryByText(/Added \(/)).not.toBeInTheDocument();
  });

  it('shows loading and current errors, then retries through Refresh', async () => {
    // Failed requests report operational feedback and remain manually retryable.
    const first = deferred<AuditLogResponse>();
    vi.mocked(pipelineVersionsApi.audit).mockReturnValueOnce(first.promise);
    renderPage();
    expect(await screen.findByText('Loading audit trail…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
    await act(async () => first.reject(new Error('Audit unavailable')));
    expect(screen.getByText('Audit unavailable')).toBeInTheDocument();
    expect(toast.error).toHaveBeenCalledExactlyOnceWith('Audit unavailable');
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(response(auditEntries));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await screen.findByText('Latest auto save');
    expect(screen.queryByText('Audit unavailable')).not.toBeInTheDocument();
  });

  it('retains previous history and summaries when a refresh fails with an empty message', async () => {
    // Failure clears loading but preserves the last successful data and fallback message.
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValueOnce(response(auditEntries)).mockRejectedValueOnce(new Error(''));
    renderPage();
    await screen.findByText('Latest auto save');
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await screen.findByText('Failed to load audit trail');
    expect(screen.getByText('Latest auto save')).toBeInTheDocument();
    expect(screen.getByText('Saves')).toBeInTheDocument();
    expect(toast.error).toHaveBeenCalledWith('Failed to load audit trail');
  });

  it.each(['resolve', 'reject'] as const)('ignores a stale request that later %ss', async outcome => {
    // Outdated success and error results must not overwrite the newest page or toast.
    const old = deferred<AuditLogResponse>();
    const current = deferred<AuditLogResponse>();
    vi.mocked(pipelineVersionsApi.audit).mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
    renderPage();
    await screen.findByText('Loading audit trail…');
    fireEvent.click(screen.getByRole('button', { name: '25' }));
    await act(async () => current.resolve(response([auditEntries[1]!])));
    expect(screen.getByText('Manual midpoint save')).toBeInTheDocument();
    await act(async () => {
      if (outcome === 'resolve') old.resolve(response(auditEntries));
      else old.reject(new Error('Stale failure'));
    });
    expect(screen.queryByText('Latest auto save')).not.toBeInTheDocument();
    expect(screen.getByText('Manual midpoint save')).toBeInTheDocument();
    expect(toast.error).not.toHaveBeenCalled();
  });

  it('keeps the current request loading when an older request finishes first', async () => {
    // A stale finally block must not re-enable refresh while the latest call is pending.
    const old = deferred<AuditLogResponse>();
    const current = deferred<AuditLogResponse>();
    vi.mocked(pipelineVersionsApi.audit).mockReturnValueOnce(old.promise).mockReturnValueOnce(current.promise);
    renderPage();
    await screen.findByText('Loading audit trail…');
    fireEvent.click(screen.getByRole('button', { name: '25' }));
    await act(async () => old.resolve(response(auditEntries)));
    expect(screen.getByText('Loading audit trail…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
    await act(async () => current.resolve(response([])));
    expect(screen.getByRole('button', { name: 'Refresh' })).toBeEnabled();
  });

  it('keeps summaries and facets during refresh while resetting expanded row state', async () => {
    // Loading unmounts the keyed rows but does not clear the last successful summary.
    const refresh = deferred<AuditLogResponse>();
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValueOnce(response(auditEntries)).mockReturnValueOnce(refresh.promise);
    renderPage();
    await screen.findByText('Latest auto save');
    fireEvent.click(screen.getByRole('button', { name: /Latest auto save/ }));
    expect(screen.getByText('node-c')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(screen.getByText('Loading audit trail…')).toBeInTheDocument();
    expect(screen.getByText('Saves')).toBeInTheDocument();
    expect(screen.getByLabelText('Actor')).toBeEnabled();
    expect(screen.getByRole('option', { name: 'user #9' })).toBeInTheDocument();
    await act(async () => refresh.resolve(response(auditEntries)));
    expect(screen.getByText('Latest auto save')).toBeInTheDocument();
    expect(screen.queryByText('node-c')).not.toBeInTheDocument();
  });

  it('defaults missing legacy facets and totals without discarding returned rows', async () => {
    // Older payloads retain the page's empty-facet and zero-total fallbacks.
    const legacy = { dataset_source_id: 'dataset-alpha', entries: auditEntries } as AuditLogResponse;
    vi.mocked(pipelineVersionsApi.audit).mockResolvedValue(legacy);
    renderPage();
    await screen.findByText('Latest auto save');
    expect(within(screen.getByLabelText('Actor')).getAllByRole('option')).toHaveLength(1);
    expect(within(screen.getByLabelText('Action kind')).getAllByRole('option')).toHaveLength(1);
    expect(screen.getByText(/Showing 3 of 0 saves for Dataset Alpha\./)).toBeInTheDocument();
  });
});
