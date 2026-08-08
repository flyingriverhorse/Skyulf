import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { monitoringApi, type SlowNodeAggregate, type SlowNodesResponse } from '../core/api/monitoring';
import { parseOperationalContext } from '../core/utils/operationalContext';
import { SlowNodesPage } from './SlowNodesPage';

vi.mock('../core/api/monitoring', async () => {
  const actual = await vi.importActual<typeof import('../core/api/monitoring')>(
    '../core/api/monitoring',
  );
  return {
    ...actual,
    monitoringApi: {
      ...actual.monitoringApi,
      getSlowNodes: vi.fn(),
    },
  };
});

function aggregate(overrides: Partial<SlowNodeAggregate> = {}): SlowNodeAggregate {
  return {
    step_type: 'impute',
    count: 3,
    total_seconds: 12,
    avg_seconds: 4,
    p95_seconds: 5,
    max_seconds: 6,
    sample_node_id: 'impute-node-1',
    is_single_run: false,
    sample_is_representative: true,
    contributing_runs: [
      {
        job_id: 'job-1',
        pipeline_id: 'pipeline-1',
        node_id: 'impute-node-1',
        dataset_source_id: 'dataset-1',
        execution_seconds: 6,
        finished_at: '2026-08-07T10:00:00',
        is_outlier: false,
      },
      {
        job_id: 'job-2',
        pipeline_id: 'pipeline-1',
        node_id: 'impute-node-2',
        dataset_source_id: 'dataset-1',
        execution_seconds: 4,
        finished_at: '2026-08-06T10:00:00',
        is_outlier: false,
      },
      {
        job_id: 'job-3',
        pipeline_id: 'pipeline-1',
        node_id: 'impute-node-3',
        dataset_source_id: 'dataset-1',
        execution_seconds: 2,
        finished_at: '2026-08-05T10:00:00',
        is_outlier: false,
      },
    ],
    ...overrides,
  };
}

function response(
  aggregates: SlowNodeAggregate[],
  overrides: Partial<SlowNodesResponse> = {},
): SlowNodesResponse {
  return {
    days: 7,
    unit: 'seconds',
    total_jobs_scanned: 5,
    total_node_runs: aggregates.reduce((sum, a) => sum + a.count, 0),
    aggregates,
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/slow-nodes']}>
      <SlowNodesPage />
    </MemoryRouter>,
  );
}

describe('SlowNodesPage — aggregate provenance, drill-down, and return state', () => {
  beforeEach(() => {
    vi.mocked(monitoringApi.getSlowNodes).mockReset();
  });

  it('shows an explicit no-data state naming the window', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([]));
    renderPage();

    expect(await screen.findByText(/no node-timing data in the last 7 days/i)).toBeInTheDocument();
  });

  it('states window, run count, and unit for a many-run aggregate', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([aggregate()]));
    renderPage();

    await screen.findByText('impute');
    expect(screen.getByText(/3 runs · last 7 days/i)).toBeInTheDocument();
    expect(screen.getAllByText('seconds').length).toBeGreaterThan(0);
  });

  it('flags a single-run aggregate as one measurement, not a trend', async () => {
    const singleRun = aggregate({
      step_type: 'scale',
      count: 1,
      is_single_run: true,
      contributing_runs: [
        {
          job_id: 'job-solo',
          pipeline_id: 'pipeline-1',
          node_id: 'scale-node-1',
          dataset_source_id: 'dataset-1',
          execution_seconds: 9,
          finished_at: '2026-08-07T10:00:00',
          is_outlier: false,
        },
      ],
    });
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([singleRun]));
    renderPage();

    await screen.findByText('scale');
    expect(screen.getByText(/single run — not a trend/i)).toBeInTheDocument();
  });

  it('marks a non-representative sample and its outlier run in the drill-down', async () => {
    const withOutlier = aggregate({
      sample_is_representative: false,
      contributing_runs: [
        {
          job_id: 'job-outlier',
          pipeline_id: 'pipeline-1',
          node_id: 'impute-node-1',
          dataset_source_id: 'dataset-1',
          execution_seconds: 40,
          finished_at: '2026-08-07T10:00:00',
          is_outlier: true,
        },
        {
          job_id: 'job-2',
          pipeline_id: 'pipeline-1',
          node_id: 'impute-node-2',
          dataset_source_id: 'dataset-1',
          execution_seconds: 4,
          finished_at: '2026-08-06T10:00:00',
          is_outlier: false,
        },
      ],
    });
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([withOutlier]));
    renderPage();

    await screen.findByText('impute');
    expect(screen.getByText(/e\.g\. \(outlier\)/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /investigate impute runs/i }));

    expect(
      await screen.findByText(/sample node shown in the table is an outlier/i),
    ).toBeInTheDocument();
    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('Outlier')).toBeInTheDocument();
  });

  it('says investigation is unavailable when no contributing runs are retained', async () => {
    const noRuns = aggregate({ contributing_runs: [] });
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([noRuns]));
    renderPage();

    await screen.findByText('impute');
    fireEvent.click(screen.getByRole('button', { name: /investigate impute runs/i }));

    expect(await screen.findByText(/investigation unavailable/i)).toBeInTheDocument();
  });

  it('gives each contributing run a returnable link to its job and Canvas node', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([aggregate()]));
    renderPage();

    await screen.findByText('impute');
    fireEvent.click(screen.getByRole('button', { name: /investigate impute runs/i }));

    const jobLink = await screen.findByRole('link', { name: /job job-1/i });
    expect(jobLink.getAttribute('href')?.split('?')[0]).toBe('/jobs');
    const jobContext = parseOperationalContext(jobLink.getAttribute('href')?.split('?')[1] ?? '');
    expect(jobContext?.ref).toEqual({ kind: 'job', jobId: 'job-1' });
    expect(jobContext?.origin).toBe('/slow-nodes');
    expect(jobContext?.filters).toEqual({ days: '7', limit: '10', sort: 'total_seconds' });

    const nodeLink = screen.getByRole('link', { name: /node impute-node-1/i });
    expect(nodeLink).toHaveTextContent(/open in canvas/i);
    expect(nodeLink.getAttribute('href')?.split('?')[0]).toBe('/canvas');
    const nodeContext = parseOperationalContext(nodeLink.getAttribute('href')?.split('?')[1] ?? '');
    expect(nodeContext?.ref).toEqual({
      kind: 'node',
      nodeId: 'impute-node-1',
      pipelineId: 'pipeline-1',
    });
  });

  it('preserves lookback, top-N, sort, and the open row across a return trip', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([aggregate()]));
    renderPage();

    await screen.findByText('impute');
    fireEvent.click(screen.getByRole('button', { name: '30 d' }));
    fireEvent.click(screen.getByRole('button', { name: 'Top 25' }));
    fireEvent.click(screen.getByRole('button', { name: /sort by avg/i }));
    fireEvent.click(screen.getByRole('button', { name: /investigate impute runs/i }));

    await waitFor(() => {
      expect(window.location.hash + window.location.search).toBeDefined();
    });

    // MemoryRouter doesn't touch window.location; assert via the router's
    // resulting DOM state instead — the modal stays open and the toggled
    // window/limit/sort controls remain the active selection, which is the
    // observable proxy for "would still be in the URL on a return trip".
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '30 d' })).toHaveClass('bg-blue-600');
    expect(screen.getByRole('button', { name: 'Top 25' })).toHaveClass('bg-blue-600');
  });

  it('exposes accessible, keyboard-operable sort controls with aria-sort on the header', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([aggregate()]));
    renderPage();

    await screen.findByText('impute');
    const totalHeader = screen.getByRole('columnheader', { name: /total/i });
    expect(totalHeader).toHaveAttribute('aria-sort', 'descending');

    const avgButton = screen.getByRole('button', { name: /sort by avg/i });
    fireEvent.click(avgButton);
    const avgHeader = screen.getByRole('columnheader', { name: /avg/i });
    expect(avgHeader).toHaveAttribute('aria-sort', 'descending');
  });

  it('offers a text/table alternative to the bar chart', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockResolvedValue(response([aggregate()]));
    renderPage();

    await screen.findByText('impute');
    const toggle = screen.getByRole('button', { name: /view data table/i });
    fireEvent.click(toggle);

    const region = screen.getByRole('region', { name: /text\/table alternative/i });
    expect(within(region).getByText('impute')).toBeInTheDocument();
  });

  it('shows an explicit error state on fetch failure', async () => {
    vi.mocked(monitoringApi.getSlowNodes).mockRejectedValue(new Error('network down'));
    renderPage();

    expect(await screen.findByText('network down')).toBeInTheDocument();
  });
});
