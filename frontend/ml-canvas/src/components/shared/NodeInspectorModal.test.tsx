import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { NodeInspectorModal } from './NodeInspectorModal';
import { monitoringApi, type NodeInspectorResponse } from '../../core/api/monitoring';

vi.mock('../../core/api/monitoring', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/monitoring')>(
    '../../core/api/monitoring',
  );
  return {
    ...actual,
    monitoringApi: {
      ...actual.monitoringApi,
      getJobNode: vi.fn(),
      getPipelineRunNode: vi.fn(),
    },
  };
});

function baseResponse(overrides: Partial<NodeInspectorResponse> = {}): NodeInspectorResponse {
  return {
    job_id: 'job-1',
    node_id: 'train-1',
    node_found: true,
    node: {
      node_id: 'train-1',
      step_type: 'training',
      label: 'Training',
      params: { algorithm: 'RandomForest' },
      upstream: [{ node_id: 'impute-1', step_type: 'simple_imputer', label: 'Simple Imputer' }],
      downstream: [],
      execution_seconds: 4.5,
      execution_status: 'success',
    },
    pipeline_id: 'preview_abc123',
    dataset_source_id: 'ds-1',
    dataset_name: 'Sales Data',
    branch_index: null,
    run_mode: 'fixed',
    model_type: 'RandomForest',
    status: 'completed',
    started_at: '2026-08-07T10:00:00',
    finished_at: '2026-08-07T10:01:00',
    is_synthetic_pipeline: true,
    can_open_in_canvas: false,
    recent_logs: [],
    ...overrides,
  };
}

function renderModal(response: NodeInspectorResponse | Error) {
  if (response instanceof Error) {
    vi.mocked(monitoringApi.getJobNode).mockRejectedValue(response);
  } else {
    vi.mocked(monitoringApi.getJobNode).mockResolvedValue(response);
  }
  const onClose = vi.fn();
  render(
    <MemoryRouter>
      <NodeInspectorModal
        isOpen
        onClose={onClose}
        target={{ kind: 'job', jobId: 'job-1' }}
        nodeId="train-1"
      />
    </MemoryRouter>,
  );
  return { onClose };
}

describe('NodeInspectorModal', () => {
  beforeEach(() => {
    vi.mocked(monitoringApi.getJobNode).mockReset();
    vi.mocked(monitoringApi.getPipelineRunNode).mockReset();
  });

  it('shows a loading state while the node detail is being fetched', () => {
    vi.mocked(monitoringApi.getJobNode).mockReturnValue(new Promise(() => {}));
    render(
      <MemoryRouter>
        <NodeInspectorModal
          isOpen
          onClose={vi.fn()}
          target={{ kind: 'job', jobId: 'job-1' }}
          nodeId="train-1"
        />
      </MemoryRouter>,
    );
    expect(screen.getByRole('status')).toHaveTextContent(/loading node detail/i);
  });

  it('renders full node detail with provenance when the node is found', async () => {
    renderModal(baseResponse());

    expect(await screen.findByText('Training')).toBeInTheDocument();
    expect(screen.getByText(/this is the graph as executed on/i)).toBeInTheDocument();
    expect(screen.getByText(/4\.50s/)).toBeInTheDocument();
    expect(screen.getByText(/simple imputer/i)).toBeInTheDocument();
    // Synthetic preview run: no canvas link offered.
    expect(screen.queryByRole('link', { name: /open in canvas/i })).not.toBeInTheDocument();
    expect(screen.getByText(/isn't a saved pipeline/i)).toBeInTheDocument();
  });

  it('renders an explicit not-found state when node_found is false', async () => {
    renderModal(
      baseResponse({
        node_found: false,
        node: null,
        node_id: 'ghost-node',
      }),
    );

    expect(
      await screen.findByText(/node not found in this job's executed graph/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/ghost-node/)).toBeInTheDocument();
    // Job-level context still renders even when the node itself is gone.
    expect(screen.getByRole('link', { name: /job job-1/i })).toBeInTheDocument();
  });

  it('renders a plain error state when the job record itself is gone', async () => {
    renderModal(new Error('Job job-1 not found'));

    expect(await screen.findByText('Job job-1 not found')).toBeInTheDocument();
  });

  it('offers a canvas link only for a genuinely saved, non-synthetic pipeline', async () => {
    renderModal(baseResponse({ pipeline_id: 'dataset-42-v3', is_synthetic_pipeline: false, can_open_in_canvas: true }));

    const canvasLink = await screen.findByRole('link', { name: /node train-1/i });
    expect(canvasLink).toHaveTextContent(/open in canvas/i);
    expect(canvasLink.getAttribute('href')?.split('?')[0]).toBe('/canvas');
  });

  it('walks to an upstream neighbour and re-fetches its detail', async () => {
    vi.mocked(monitoringApi.getJobNode).mockImplementation(async (_jobId, nodeId) => {
      if (nodeId === 'train-1') return baseResponse();
      return baseResponse({
        node_id: 'impute-1',
        node: {
          node_id: 'impute-1',
          step_type: 'simple_imputer',
          label: 'Simple Imputer',
          params: { strategy: 'mean' },
          upstream: [],
          downstream: [{ node_id: 'train-1', step_type: 'training', label: 'Training' }],
        },
      });
    });

    render(
      <MemoryRouter>
        <NodeInspectorModal
          isOpen
          onClose={vi.fn()}
          target={{ kind: 'job', jobId: 'job-1' }}
          nodeId="train-1"
        />
      </MemoryRouter>,
    );

    await screen.findByText('Training');
    fireEvent.click(screen.getByRole('button', { name: /simple imputer \(impute-1\)/i }));

    await waitFor(() => {
      expect(monitoringApi.getJobNode).toHaveBeenLastCalledWith('job-1', 'impute-1');
    });
    expect(await screen.findByText('Simple Imputer')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /training \(train-1\)/i })).toBeInTheDocument();
  });
});
