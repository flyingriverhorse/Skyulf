import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach } from 'vitest';

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

  it('gives a pipeline log with a node_id a contextual View action to the node', async () => {
    renderPage();
    await screen.findByText('node failed');

    const link = screen.getByRole('link', { name: /node node-xyz/i });
    expect(link.getAttribute('href')?.split('?')[0]).toBe('/canvas');
    const parsed = parseOperationalContext(link.getAttribute('href')?.split('?')[1] ?? '');
    expect(parsed?.ref).toEqual({ kind: 'node', nodeId: 'node-xyz', pipelineId: 'pipeline-1' });
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
    await screen.findByText('boom');

    fireEvent.click(screen.getByRole('button', { name: /export csv/i }));

    expect(createObjectURL).toHaveBeenCalled();
    const text = blobParts.flat().join('');
    expect(text).toContain('job-abc-123');
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
});
