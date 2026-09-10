import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import type { ReactNode } from 'react';

import {
  monitoringApi,
  type ErrorEvent,
  type ErrorEventSearchResponse,
  type PipelineLogSearchResponse,
  type PipelineRunLog,
} from '../core/api/monitoring';
import { parseOperationalContext } from '../core/utils/operationalContext';
import { ConfirmProvider } from '../components/shared';
import { ErrorLogPage } from './ErrorLogPage';
import { toast } from '../core/toast';

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => <>{children}</>,
  BarChart: ({ data }: { data: { hour: string; count: number }[] }) => (
    <output aria-label="Timeline buckets">{JSON.stringify(data)}</output>
  ),
  Bar: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  Cell: () => null,
}));

function deferred<T>() {
  // Let tests observe the public page while a particular request is pending.
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

vi.mock('../core/api/monitoring', async () => {
  const actual = await vi.importActual<typeof import('../core/api/monitoring')>(
    '../core/api/monitoring',
  );
  return {
    ...actual,
    monitoringApi: {
      ...actual.monitoringApi,
      getErrors: vi.fn(),
      getPipelineLogs: vi.fn(),
      getTimeline: vi.fn(),
      getGrouped: vi.fn(),
      getError: vi.fn(),
      resolveError: vi.fn(),
      unresolveError: vi.fn(),
      clearErrors: vi.fn(),
      clearPipelineLogs: vi.fn(),
      getPipelineRunNode: vi.fn(),
      getJobNode: vi.fn(),
    },
  };
});

const FACETS = {
  severities: ['critical', 'warning'] as const,
  error_types: ['ValueError', 'PipelineExecutionException'],
  job_ids: ['job-abc-123'],
};

function errorEvent(overrides: Partial<ErrorEvent> = {}): ErrorEvent {
  return {
    id: 1,
    route: '/api/pipeline/datasets/273/schema',
    error_type: 'ValueError',
    message: 'boom',
    traceback: 'Traceback...',
    job_id: 'job-abc-123',
    status_code: 500,
    created_at: '2026-08-07T10:00:00.000Z',
    resolved_at: null,
    severity: 'critical',
    ...overrides,
  };
}

function pipelineLog(overrides: Partial<PipelineRunLog> = {}): PipelineRunLog {
  return {
    id: 9,
    pipeline_id: 'pipeline-1',
    node_id: 'node-xyz',
    node_type: 'encoder',
    level: 'error',
    logger: 'skyulf',
    message: 'node failed',
    run_at: '2026-08-07T10:05:00',
    ...overrides,
  };
}

function errorsResponse(
  entries: ErrorEvent[],
  overrides: Partial<ErrorEventSearchResponse> = {},
): ErrorEventSearchResponse {
  return {
    total: entries.length,
    total_unfiltered: entries.length,
    facets: {
      severities: [...FACETS.severities],
      error_types: [...FACETS.error_types],
      job_ids: [...FACETS.job_ids],
    },
    filters: {
      since: null,
      show_resolved: false,
      severity: null,
      error_type: null,
      job_id: null,
      q: null,
    },
    entries,
    ...overrides,
  };
}

function pipelineResponse(
  entries: PipelineRunLog[],
  overrides: Partial<PipelineLogSearchResponse> = {},
): PipelineLogSearchResponse {
  return {
    total: entries.length,
    total_unfiltered: entries.length,
    facets: {
      levels: ['error', 'warning'],
      node_types: ['encoder'],
      pipeline_ids: ['pipeline-1'],
      node_ids: ['node-xyz'],
    },
    filters: {
      since: null,
      pipeline_id: null,
      level: null,
      node_type: null,
      node_id: null,
      q: null,
    },
    entries,
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter>
      <ConfirmProvider>
        <ErrorLogPage />
      </ConfirmProvider>
    </MemoryRouter>,
  );
}

describe('ErrorLogPage — server-side search, facets, links, and export', () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });
  beforeEach(() => {
    vi.mocked(monitoringApi.getErrors).mockReset().mockResolvedValue(errorsResponse([errorEvent()]));
    vi.mocked(monitoringApi.getPipelineLogs)
      .mockReset()
      .mockResolvedValue(pipelineResponse([pipelineLog()]));
    vi.mocked(monitoringApi.getTimeline).mockReset().mockResolvedValue([]);
    vi.mocked(monitoringApi.getGrouped).mockReset().mockResolvedValue([]);
    vi.mocked(monitoringApi.getError).mockReset();
    vi.mocked(monitoringApi.resolveError).mockReset();
    vi.mocked(monitoringApi.unresolveError).mockReset();
    vi.mocked(monitoringApi.clearErrors).mockReset();
    vi.mocked(monitoringApi.clearPipelineLogs).mockReset();
  });

  it('sends an exact HTTP job_id typed into the generic search as the server-side `q` filter', async () => {
    renderPage();
    await screen.findByText('boom');

    fireEvent.change(screen.getByPlaceholderText('Search errors, job id, node id…'), {
      target: { value: 'job-abc-123' },
    });

    await waitFor(() => {
      expect(monitoringApi.getErrors).toHaveBeenLastCalledWith(
        500,
        expect.anything(),
        false,
        expect.objectContaining({ q: 'job-abc-123' }),
      );
    });
    // Pipeline logs are searched with the same generic query, so an exact
    // pipeline node_id typed into the same box is still found.
    await waitFor(() => {
      expect(monitoringApi.getPipelineLogs).toHaveBeenLastCalledWith(
        200,
        expect.anything(),
        undefined,
        expect.objectContaining({ q: 'job-abc-123' }),
      );
    });
  });

  it('sends an exact pipeline node_id typed into the generic search as the server-side `q` filter', async () => {
    renderPage();
    await screen.findByText('node failed');

    fireEvent.change(screen.getByPlaceholderText('Search errors, job id, node id…'), {
      target: { value: 'node-xyz' },
    });

    await waitFor(() => {
      expect(monitoringApi.getPipelineLogs).toHaveBeenLastCalledWith(
        200,
        expect.anything(),
        undefined,
        expect.objectContaining({ q: 'node-xyz' }),
      );
    });
  });

  it('applies the severity facet server-side rather than filtering the loaded page', async () => {
    renderPage();
    await screen.findByText('boom');

    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([]));
    fireEvent.change(screen.getByDisplayValue('All severities'), { target: { value: 'warning' } });

    await waitFor(() => {
      expect(monitoringApi.getErrors).toHaveBeenLastCalledWith(
        500,
        expect.anything(),
        false,
        expect.objectContaining({ severity: 'warning' }),
      );
    });
  });

  it('applies the job id facet server-side', async () => {
    renderPage();
    await screen.findByText('boom');

    fireEvent.change(screen.getByDisplayValue('All job IDs'), { target: { value: 'job-abc-123' } });

    await waitFor(() => {
      expect(monitoringApi.getErrors).toHaveBeenLastCalledWith(
        500,
        expect.anything(),
        false,
        expect.objectContaining({ jobId: 'job-abc-123' }),
      );
    });
  });

  it('applies the node id facet server-side to pipeline logs', async () => {
    renderPage();
    await screen.findByText('node failed');

    fireEvent.change(screen.getByDisplayValue('All node IDs'), { target: { value: 'node-xyz' } });

    await waitFor(() => {
      expect(monitoringApi.getPipelineLogs).toHaveBeenLastCalledWith(
        200,
        expect.anything(),
        undefined,
        expect.objectContaining({ nodeId: 'node-xyz' }),
      );
    });
  });

  it('gives an HTTP event with a job_id a contextual View action to the job', async () => {
    renderPage();
    await screen.findByText('boom');

    const link = screen.getByRole('link', { name: /job job-abc-123/i });
    expect(link.getAttribute('href')?.split('?')[0]).toBe('/jobs');
    const parsed = parseOperationalContext(link.getAttribute('href')?.split('?')[1] ?? '');
    expect(parsed?.ref).toEqual({ kind: 'job', jobId: 'job-abc-123' });
    expect(parsed?.origin).toBe('/errors');
  });

  it('gives a pipeline log with a node_id a contextual action to inspect the node', async () => {
    vi.mocked(monitoringApi.getPipelineRunNode).mockResolvedValue({
      job_id: 'job-9',
      node_id: 'node-xyz',
      node_found: true,
      node: {
        node_id: 'node-xyz',
        step_type: 'simple_imputer',
        label: 'Simple Imputer',
        params: { strategy: 'mean' },
        upstream: [],
        downstream: [],
      },
      pipeline_id: 'pipeline-1',
      dataset_source_id: 'ds-1',
      dataset_name: 'Dataset 1',
      run_mode: 'fixed',
      model_type: 'RandomForest',
      status: 'completed',
      is_synthetic_pipeline: false,
      can_open_in_canvas: true,
      recent_logs: [],
    });
    renderPage();
    await screen.findByText('node failed');

    const trigger = screen.getByRole('button', { name: /node node-xyz/i });
    fireEvent.click(trigger);

    expect(await screen.findByText('Simple Imputer')).toBeInTheDocument();
    expect(monitoringApi.getPipelineRunNode).toHaveBeenCalledWith('pipeline-1', 'node-xyz');
  });

  it('tells the investigator explicitly when no target is available', async () => {
    const { job_id: _jobId, ...eventWithoutJob } = errorEvent({ id: 2 });
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(
      errorsResponse([eventWithoutJob as ErrorEvent]),
    );
    renderPage();
    await screen.findByText('boom');

    expect(screen.getByText('No target available')).toBeInTheDocument();
  });

  it('distinguishes "no history" from "no match" for the empty state', async () => {
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([], { total_unfiltered: 0 }));
    vi.mocked(monitoringApi.getPipelineLogs).mockResolvedValue(pipelineResponse([], { total_unfiltered: 0 }));
    renderPage();

    fireEvent.change(await screen.findByPlaceholderText('Search errors, job id, node id…'), {
      target: { value: 'anything' },
    });

    await screen.findByText(/no error events have been recorded yet/i);
  });

  it('reports "no match" (not "no history") when filters exclude an otherwise non-empty history', async () => {
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([], { total_unfiltered: 42 }));
    vi.mocked(monitoringApi.getPipelineLogs).mockResolvedValue(pipelineResponse([], { total_unfiltered: 0 }));
    renderPage();

    fireEvent.change(await screen.findByPlaceholderText('Search errors, job id, node id…'), {
      target: { value: 'no-such-error' },
    });

    await screen.findByText(/no events match the current search\/facets out of 42 recorded/i);
  });

  it('exports the currently visible (filtered) rows as CSV', async () => {
    // Quoting preserves delimiters, quotes and newlines in the downloaded data.
    const exportedEvent = errorEvent({ message: 'line "quoted",\nnext' });
    delete exportedEvent.job_id;
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([exportedEvent]));
    const createObjectURL = vi.fn().mockReturnValue('blob:mock');
    const revokeObjectURL = vi.fn();
    Object.defineProperty(URL, 'createObjectURL', { value: createObjectURL, writable: true });
    Object.defineProperty(URL, 'revokeObjectURL', { value: revokeObjectURL, writable: true });
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    const blobParts: BlobPart[][] = [];
    const RealBlob = globalThis.Blob;
    const blobSpy = vi.spyOn(globalThis, 'Blob').mockImplementation(
      class extends RealBlob {
        constructor(parts?: BlobPart[], options?: BlobPropertyBag) {
          super(parts, options);
          blobParts.push(parts ?? []);
        }
      } as unknown as typeof Blob,
    );

    renderPage();
    await screen.findByText('line "quoted", next');

    fireEvent.click(screen.getByRole('button', { name: /export csv/i }));

    expect(createObjectURL).toHaveBeenCalled();
    const text = blobParts.flat().join('');
    expect(text).toContain('"line ""quoted"",\nnext"');
    expect(text).toContain('"/api/pipeline/datasets/273/schema","",');
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:mock');
    expect(text).toContain('critical');
    expect(clickSpy).toHaveBeenCalled();

    clickSpy.mockRestore();
    blobSpy.mockRestore();
  });

  it('still finds and resolves an event by id after a facet-filtered reload', async () => {
    renderPage();
    await screen.findByText('boom');

    const resolved = errorEvent({ resolved_at: '2026-08-07T11:00:00.000Z' });
    vi.mocked(monitoringApi.resolveError).mockResolvedValue(resolved);

    fireEvent.click(screen.getByTitle('Mark resolved'));

    await waitFor(() => expect(monitoringApi.resolveError).toHaveBeenCalledWith(1));
    await screen.findByTitle('Reopen');
  });

  it('finishes HTTP loading before pipeline logs arrive and preserves the server row order', async () => {
    // A slow pipeline request must not block usable HTTP results.
    const http = deferred<ErrorEventSearchResponse>();
    const pipeline = deferred<PipelineLogSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValue(http.promise);
    vi.mocked(monitoringApi.getPipelineLogs).mockReturnValue(pipeline.promise);
    renderPage();
    expect(screen.getByText('Loading error events…')).toBeInTheDocument();
    expect(monitoringApi.getPipelineLogs).not.toHaveBeenCalled();
    await act(async () => http.resolve(errorsResponse([errorEvent({ id: 2, message: 'first HTTP' }), errorEvent()])));
    expect(screen.queryByText('Loading error events…')).not.toBeInTheDocument();
    expect(screen.getByText('first HTTP')).toBeInTheDocument();
    await act(async () => pipeline.resolve(pipelineResponse([pipelineLog()])));
    const rows = screen.getAllByRole('row');
    expect(rows.slice(1).map(row => within(row).getAllByRole('cell')[4]!.textContent))
      .toEqual(['node failed', 'first HTTP', 'boom']);
    expect(monitoringApi.getTimeline).toHaveBeenCalledWith(24);
    expect(monitoringApi.getGrouped).toHaveBeenCalledWith();
  });

  it('keeps the latest search results when an older request finishes last', async () => {
    // OC-221: results and diagnostic context must describe the same current search.
    renderPage();
    await screen.findByText('boom');
    const older = deferred<ErrorEventSearchResponse>();
    const newer = deferred<ErrorEventSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
    const search = screen.getByPlaceholderText('Search errors, job id, node id…');
    fireEvent.change(search, { target: { value: 'older' } });
    fireEvent.change(search, { target: { value: 'newer' } });
    await act(async () => newer.resolve(errorsResponse([errorEvent({ message: 'newer result' })])));
    expect(screen.getByText('newer result')).toBeInTheDocument();
    await act(async () => older.resolve(errorsResponse([errorEvent({ message: 'older result' })])));
    expect(search).toHaveValue('newer');
    expect(screen.queryByText('older result')).not.toBeInTheDocument();
    expect(screen.getByText('newer result')).toBeInTheDocument();
    expect(monitoringApi.getPipelineLogs).toHaveBeenLastCalledWith(
      200, expect.anything(), undefined, { q: 'newer' },
    );
  });

  it('keeps loading the latest search when an older success settles first', async () => {
    // An obsolete request must neither publish rows nor settle the active loading state.
    renderPage();
    await screen.findByText('boom');
    const older = deferred<ErrorEventSearchResponse>();
    const newer = deferred<ErrorEventSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
    const search = screen.getByPlaceholderText('Search errors, job id, node id…');
    fireEvent.change(search, { target: { value: 'older' } });
    fireEvent.change(search, { target: { value: 'newer' } });
    await act(async () => older.resolve(errorsResponse([errorEvent({ message: 'older result' })])));
    expect(screen.getByText('Loading error events…')).toBeInTheDocument();
    expect(monitoringApi.getPipelineLogs).toHaveBeenCalledTimes(1);
    await act(async () => newer.resolve(errorsResponse([errorEvent({ message: 'newer result' })])));
    expect(screen.getByText('newer result')).toBeInTheDocument();
    expect(screen.queryByText('Loading error events…')).not.toBeInTheDocument();
  });

  it.each(['pending', 'complete'] as const)(
    'ignores a stale rejection while the latest search is %s', async (latestState) => {
      // Old failures cannot hide current results or end the current request's loading state.
      renderPage();
      await screen.findByText('boom');
      const older = deferred<ErrorEventSearchResponse>();
      const newer = deferred<ErrorEventSearchResponse>();
      vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
      const search = screen.getByPlaceholderText('Search errors, job id, node id…');
      fireEvent.change(search, { target: { value: 'older' } });
      fireEvent.change(search, { target: { value: 'newer' } });
      if (latestState === 'complete') {
        await act(async () => newer.resolve(errorsResponse([errorEvent({ message: 'newer result' })])));
      }
      await act(async () => older.reject(new Error('obsolete failure')));
      expect(screen.queryByText('Could not reach the backend. Is the server running?')).not.toBeInTheDocument();
      if (latestState === 'pending') {
        expect(screen.getByText('Loading error events…')).toBeInTheDocument();
        await act(async () => newer.resolve(errorsResponse([errorEvent({ message: 'newer result' })])));
      }
      expect(screen.getByText('newer result')).toBeInTheDocument();
    },
  );

  it('keeps current facets, totals and issue groups when a stale HTTP bundle resolves', async () => {
    // Request ownership covers the full HTTP snapshot, not just its event rows.
    renderPage();
    await screen.findByText('boom');
    const older = deferred<ErrorEventSearchResponse>();
    const newer = deferred<ErrorEventSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
    vi.mocked(monitoringApi.getGrouped).mockResolvedValueOnce([{
      error_type: 'OldError', route: '/old', count: 2, sample_id: 99, first_seen: '', last_seen: '',
    }]).mockResolvedValueOnce([]);
    const search = screen.getByPlaceholderText('Search errors, job id, node id…');
    fireEvent.change(search, { target: { value: 'older' } });
    fireEvent.change(search, { target: { value: 'newer' } });
    await act(async () => newer.resolve(errorsResponse([errorEvent()], { total: 1, total_unfiltered: 20 })));
    await act(async () => older.resolve(errorsResponse([errorEvent()], {
      total: 2, total_unfiltered: 10,
      facets: { severities: ['critical'], error_types: ['OldError'], job_ids: [] },
    })));
    expect(screen.getByText('1 of 20 HTTP events match')).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'OldError' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Issues (1)' }));
    expect(screen.queryByText('OldError')).not.toBeInTheDocument();
  });

  it('supersedes an earlier refresh even when the filters do not change', async () => {
    // Refresh creates new request ownership independently of filter identity.
    renderPage();
    await screen.findByText('boom');
    const older = deferred<ErrorEventSearchResponse>();
    const newer = deferred<ErrorEventSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    await act(async () => newer.resolve(errorsResponse([errorEvent({ message: 'new refresh' })])));
    await act(async () => older.resolve(errorsResponse([errorEvent({ message: 'old refresh' })])));
    expect(screen.getByText('new refresh')).toBeInTheDocument();
    expect(screen.queryByText('old refresh')).not.toBeInTheDocument();
    expect(monitoringApi.getErrors).toHaveBeenCalledTimes(3);
    expect(monitoringApi.getPipelineLogs).toHaveBeenCalledTimes(2);
  });

  it('keeps the latest pipeline rows and node facets when old logs arrive after a filter change', async () => {
    // Pipeline follow-ups belong to the HTTP generation which started them.
    const older = deferred<PipelineLogSearchResponse>();
    const newer = deferred<PipelineLogSearchResponse>();
    vi.mocked(monitoringApi.getPipelineLogs).mockReturnValueOnce(older.promise).mockReturnValueOnce(newer.promise);
    renderPage();
    await screen.findByText('boom');
    fireEvent.change(screen.getByPlaceholderText('Search errors, job id, node id…'), { target: { value: 'newer' } });
    await screen.findByText('boom');
    await act(async () => newer.resolve(pipelineResponse([pipelineLog({ message: 'new pipeline' })])));
    await act(async () => older.resolve(pipelineResponse([pipelineLog({ message: 'old pipeline' })], {
      facets: { levels: ['error'], node_types: ['old'], pipeline_ids: [], node_ids: ['old-node'] },
    })));
    expect(screen.getByText('new pipeline')).toBeInTheDocument();
    expect(screen.queryByText('old pipeline')).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'old-node' })).not.toBeInTheDocument();
  });

  it('invalidates pending pipeline logs as soon as a new HTTP load starts', async () => {
    // A newer main-request failure must not reveal pipeline data from an obsolete query.
    const older = deferred<PipelineLogSearchResponse>();
    vi.mocked(monitoringApi.getPipelineLogs).mockReturnValueOnce(older.promise);
    renderPage();
    await screen.findByText('boom');
    vi.mocked(monitoringApi.getErrors).mockRejectedValueOnce(new Error('current failure'));
    fireEvent.change(screen.getByPlaceholderText('Search errors, job id, node id…'), { target: { value: 'newer' } });
    await screen.findByText('Could not reach the backend. Is the server running?');
    await act(async () => older.resolve(pipelineResponse([pipelineLog({ message: 'old pipeline' })])));
    expect(screen.getByRole('button', { name: 'Events (1)' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Clear pipeline' })).not.toBeInTheDocument();
  });

  it('ignores an obsolete pipeline rejection without affecting current HTTP results', async () => {
    // Superseded pipeline failures should not create diagnostics for the active query.
    const older = deferred<PipelineLogSearchResponse>();
    vi.mocked(monitoringApi.getPipelineLogs).mockReturnValueOnce(older.promise);
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});
    renderPage();
    await screen.findByText('boom');
    fireEvent.change(screen.getByPlaceholderText('Search errors, job id, node id…'), { target: { value: 'newer' } });
    await screen.findByText('node failed');
    await act(async () => older.reject(new Error('obsolete pipeline failure')));
    expect(screen.getByText('boom')).toBeInTheDocument();
    expect(screen.getByText('node failed')).toBeInTheDocument();
    expect(debug).not.toHaveBeenCalled();
  });

  it('does not start a pipeline follow-up when HTTP completes after unmount', async () => {
    // Leaving Error Log invalidates pending HTTP work before it can launch more requests.
    const pending = deferred<ErrorEventSearchResponse>();
    vi.mocked(monitoringApi.getErrors).mockReturnValueOnce(pending.promise);
    const page = renderPage();
    page.unmount();
    await act(async () => pending.resolve(errorsResponse([errorEvent()])));
    expect(monitoringApi.getPipelineLogs).not.toHaveBeenCalled();
  });

  it('ignores a pending pipeline failure after unmount', async () => {
    // The detached pipeline continuation shares the same unmount invalidation.
    const pending = deferred<PipelineLogSearchResponse>();
    vi.mocked(monitoringApi.getPipelineLogs).mockReturnValueOnce(pending.promise);
    const debug = vi.spyOn(console, 'debug').mockImplementation(() => {});
    const page = renderPage();
    await screen.findByText('boom');
    page.unmount();
    await act(async () => pending.reject(new Error('unmounted pipeline failure')));
    expect(debug).not.toHaveBeenCalled();
  });

  it('retries a failed main load on refresh and tolerates pipeline failure separately', async () => {
    // Main request failures show a page error; pipeline failures leave HTTP results usable.
    vi.mocked(monitoringApi.getErrors).mockRejectedValueOnce(new Error('offline'));
    vi.mocked(monitoringApi.getPipelineLogs).mockRejectedValue(new Error('pipeline offline'));
    vi.spyOn(console, 'debug').mockImplementation(() => {});
    renderPage();
    expect(await screen.findByText('Could not reach the backend. Is the server running?')).toBeInTheDocument();
    expect(monitoringApi.getPipelineLogs).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByText('boom')).toBeInTheDocument();
    expect(screen.queryByText('node failed')).not.toBeInTheDocument();
    expect(monitoringApi.getErrors).toHaveBeenCalledTimes(2);
  });

  it('composes facets, all-time scope and resolved state in requests and diagnostic links', async () => {
    // Exact typed filters and navigation scope survive independent control changes.
    renderPage();
    await screen.findByText('boom');
    fireEvent.click(screen.getByRole('button', { name: 'All' }));
    fireEvent.click(screen.getByRole('button', { name: 'Show resolved' }));
    fireEvent.change(screen.getByLabelText('All severities'), { target: { value: 'critical' } });
    fireEvent.change(screen.getByLabelText('All error types'), { target: { value: 'ValueError' } });
    fireEvent.change(screen.getByLabelText('All job IDs'), { target: { value: 'job-abc-123' } });
    fireEvent.change(screen.getByLabelText('All node IDs'), { target: { value: 'node-xyz' } });
    await screen.findByText('boom');
    expect(monitoringApi.getErrors).toHaveBeenLastCalledWith(500, undefined, true, {
      severity: 'critical', errorType: 'ValueError', jobId: 'job-abc-123',
    });
    expect(monitoringApi.getPipelineLogs).toHaveBeenLastCalledWith(200, undefined, undefined, {
      level: 'error', nodeId: 'node-xyz',
    });
    const link = screen.getByRole('link', { name: /job job-abc-123/i });
    const context = parseOperationalContext(link.getAttribute('href')?.split('?')[1] ?? '');
    expect(context?.timeRange).toBe('all');
    expect(context?.filters).toEqual({
      showResolved: 'true', severity: 'critical', errorType: 'ValueError', jobId: 'job-abc-123', nodeId: 'node-xyz',
    });
  });

  it.each([['1h', 3600000], ['6h', 21600000], ['24h', 86400000], ['7d', 604800000]] as const)(
    'sends the %s time range as an ISO lower bound', async (label, milliseconds) => {
      // Relative ranges remain anchored to the request time, not the first render.
      const now = new Date('2026-09-09T12:00:00Z');
      vi.spyOn(Date, 'now').mockReturnValue(now.getTime());
      renderPage();
      await screen.findByText('boom');
      fireEvent.click(screen.getByRole('button', { name: label }));
      await waitFor(() => expect(monitoringApi.getErrors).toHaveBeenLastCalledWith(
        500, new Date(now.getTime() - milliseconds).toISOString(), false, {},
      ));
    },
  );

  it('groups pipeline errors ahead of HTTP issues and loads the requested sample', async () => {
    // Issue counts group error-level logs only, while samples come from the detail endpoint.
    vi.mocked(monitoringApi.getPipelineLogs).mockResolvedValue(pipelineResponse([
      pipelineLog(), pipelineLog({ id: 10 }), pipelineLog({ id: 11, level: 'warning' }),
    ]));
    vi.mocked(monitoringApi.getGrouped).mockResolvedValue([{
      error_type: 'ValueError', route: '/route', count: 5, sample_id: 42,
      first_seen: '2026-09-09T09:00:00Z', last_seen: '2026-09-09T10:00:00Z',
    }]);
    vi.mocked(monitoringApi.getError).mockResolvedValue(errorEvent({ id: 42, traceback: 'sample traceback' }));
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Issues (2)' }));
    const rows = screen.getAllByRole('row').slice(1);
    expect(within(rows[0]!).getByText('encoder')).toBeInTheDocument();
    expect(within(rows[0]!).getByText('2')).toBeInTheDocument();
    expect(within(rows[1]!).getByText('5')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'View sample' }));
    expect(await screen.findByText('sample traceback')).toBeInTheDocument();
    expect(monitoringApi.getError).toHaveBeenCalledWith(42);
  });

  it('shows empty event and issue states independently', async () => {
    // Empty history retains both available views and their distinct explanations.
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([]));
    vi.mocked(monitoringApi.getPipelineLogs).mockResolvedValue(pipelineResponse([]));
    renderPage();
    expect(await screen.findByText('No errors recorded')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Issues (0)' }));
    expect(screen.getByText('No open issues')).toBeInTheDocument();
  });

  it('expands details, copies the stable ID for 1500ms, and opens the full traceback', async () => {
    // Row previews truncate long traces without losing the full modal diagnostic.
    const traceback = 'x'.repeat(810);
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([errorEvent({ traceback })]));
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText } });
    renderPage();
    fireEvent.click(await screen.findByText('boom'));
    expect(screen.getByText(/click Traceback for full output/).textContent).toBe(`${'x'.repeat(800)}\n…(click Traceback for full output)`);
    vi.useFakeTimers();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy diagnostic ID' })));
    expect(writeText).toHaveBeenCalledWith('1');
    expect(screen.getByRole('button', { name: 'Diagnostic ID copied' })).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1500));
    expect(screen.getByRole('button', { name: 'Copy diagnostic ID' })).toBeInTheDocument();
    vi.useRealTimers();
    fireEvent.click(screen.getByRole('button', { name: 'Traceback' }));
    expect(screen.getByText(traceback)).toBeInTheDocument();
    // Clicking the panel retains it, clicking its backdrop closes it.
    fireEvent.click(screen.getByText(traceback));
    expect(screen.getByText(traceback)).toBeInTheDocument();
    fireEvent.click(screen.getByText(traceback).closest('.fixed')!);
    expect(screen.queryByText(traceback)).not.toBeInTheDocument();
  });

  it('keeps a diagnostic ID selectable when clipboard writing fails', async () => {
    // Clipboard permission failures do not prevent reading diagnostics.
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true, value: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
    });
    renderPage();
    fireEvent.click(await screen.findByText('node failed'));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Copy diagnostic ID' })));
    expect(screen.getByRole('button', { name: 'Copy diagnostic ID' })).toHaveTextContent('9');
    expect(screen.getByText('Pipeline:')).toBeInTheDocument();
  });

  it('reopens a resolved event without reloading either data source', async () => {
    // Resolution replaces the returned row in place even in the unresolved default view.
    vi.mocked(monitoringApi.getErrors).mockResolvedValue(errorsResponse([errorEvent({ resolved_at: '2026-09-09T10:00:00Z' })]));
    vi.mocked(monitoringApi.unresolveError).mockResolvedValue(errorEvent());
    renderPage();
    fireEvent.click(await screen.findByTitle('Reopen'));
    expect(await screen.findByTitle('Mark resolved')).toBeInTheDocument();
    expect(monitoringApi.unresolveError).toHaveBeenCalledWith(1);
    expect(monitoringApi.getErrors).toHaveBeenCalledTimes(1);
    expect(monitoringApi.getPipelineLogs).toHaveBeenCalledTimes(1);
  });

  it('cancels HTTP deletion before confirming and disables clear while deletion is pending', async () => {
    // The confirmation protects HTTP deletion, and its pending state prevents repeated clearing.
    const deletion = deferred<{ deleted: number }>();
    vi.mocked(monitoringApi.clearErrors).mockReturnValue(deletion.promise);
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Clear HTTP' }));
    expect(screen.getByText('Delete all 1 error events? This cannot be undone.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(monitoringApi.clearErrors).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Clear HTTP' }));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Delete all' })));
    expect(screen.getByRole('button', { name: 'Clear HTTP' })).toBeDisabled();
    await act(async () => deletion.resolve({ deleted: 1 }));
    expect(screen.queryByText('boom')).not.toBeInTheDocument();
    expect(screen.getByText('node failed')).toBeInTheDocument();
    expect(monitoringApi.clearErrors).toHaveBeenCalledTimes(1);
  });

  it('reports pipeline-clear and sample failures through the notification API', async () => {
    // Failed actions retain the current data and deliver their established notification text.
    const notify = vi.spyOn(toast, 'error');
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.mocked(monitoringApi.clearPipelineLogs).mockRejectedValueOnce(new Error('failed')).mockResolvedValueOnce(undefined);
    vi.mocked(monitoringApi.getGrouped).mockResolvedValue([{
      error_type: 'ValueError', route: '/route', count: 1, sample_id: 42,
      first_seen: '', last_seen: '',
    }]);
    vi.mocked(monitoringApi.getError).mockRejectedValue(new Error('missing'));
    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: 'Clear pipeline' }));
    await waitFor(() => expect(notify).toHaveBeenCalledWith('Failed to clear pipeline logs', 'Please try again.'));
    expect(screen.getByText('node failed')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Clear pipeline' }));
    await waitFor(() => expect(screen.queryByText('node failed')).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole('button', { name: 'Issues (1)' }));
    fireEvent.click(screen.getByRole('button', { name: 'View sample' }));
    await waitFor(() => expect(notify).toHaveBeenCalledWith('Failed to load error sample', 'Please try again.'));
  });

  it('merges UTC HTTP counts and local pipeline timestamps into 24 hourly chart buckets', async () => {
    // Chart bucketing retains its timezone boundary and includes every pipeline level.
    vi.useFakeTimers({ toFake: ['Date'] });
    vi.setSystemTime(new Date('2026-09-09T12:30:00Z'));
    const local = new Date();
    const pad = (value: number) => String(value).padStart(2, '0');
    const prefix = `${local.getFullYear()}-${pad(local.getMonth() + 1)}-${pad(local.getDate())}T${pad(local.getHours())}`;
    vi.mocked(monitoringApi.getTimeline).mockResolvedValue([{ hour: '2026-09-09T12:00', count: 3 }, { hour: 'invalid', count: 99 }]);
    vi.mocked(monitoringApi.getPipelineLogs).mockResolvedValue(pipelineResponse([
      pipelineLog({ run_at: `${prefix}:05:00`, level: 'warning' }),
      pipelineLog({ id: 10, run_at: null }),
      pipelineLog({ id: 11, run_at: '2000-01-01T12:00:00' }),
    ]));
    renderPage();
    await screen.findByText('Events (4)');
    const buckets = JSON.parse(screen.getByLabelText('Timeline buckets').textContent!) as { hour: string; count: number }[];
    expect(buckets).toHaveLength(24);
    expect(buckets.at(-1)).toMatchObject({ hour: `${prefix}:00:00`, count: 4 });
    expect(buckets.reduce((sum, bucket) => sum + bucket.count, 0)).toBe(4);
  });
});
