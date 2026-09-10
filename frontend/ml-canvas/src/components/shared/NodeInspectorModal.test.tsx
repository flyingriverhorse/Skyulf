import { act, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { StrictMode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { NodeInspectorModal, type NodeInspectorModalProps } from './NodeInspectorModal';
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

/** Control response ordering without timers or replacing the real modal. */
function deferredNode() {
  let resolve!: (response: NodeInspectorResponse) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<NodeInspectorResponse>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

/** Keep the displayed identity and the response's navigation identity consistent. */
function nodeResponse(nodeId: string, label: string): NodeInspectorResponse {
  return baseResponse({
    node_id: nodeId,
    node: { node_id: nodeId, label, step_type: 'training', params: {}, upstream: [], downstream: [] },
  });
}

/** Drive selection and visibility through the modal's public props. */
function mountInspector(overrides: Partial<NodeInspectorModalProps> = {}) {
  const props: NodeInspectorModalProps = {
    isOpen: true, onClose: vi.fn(), target: { kind: 'job', jobId: 'job-1' }, nodeId: 'first', ...overrides,
  };
  const view = render(<MemoryRouter><NodeInspectorModal {...props} /></MemoryRouter>);
  return { ...view, update: (updates: Partial<NodeInspectorModalProps>) => {
    Object.assign(props, updates);
    view.rerender(<MemoryRouter><NodeInspectorModal {...props} /></MemoryRouter>);
  } };
}

describe('NodeInspectorModal', () => {
  beforeEach(() => {
    vi.mocked(monitoringApi.getJobNode).mockReset();
    vi.mocked(monitoringApi.getPipelineRunNode).mockReset();
  });

  it('uses pipeline-run requests and preserves zero values and repeated logs', async () => {
    // Historical context must retain valid zero metadata and backend log ordering.
    vi.mocked(monitoringApi.getPipelineRunNode).mockResolvedValue(baseResponse({
      branch_index: 0, finished_at: 'unparsed timestamp',
      node: { node_id: 'train-1', step_type: 'training', label: 'Zero time', params: {}, upstream: [], downstream: [], execution_seconds: 0 },
      recent_logs: [{ level: 'warning', message: 'Repeated' }, { level: 'warning', message: 'Repeated' }] as NodeInspectorResponse['recent_logs'],
    }));
    const onClose = vi.fn();
    render(<MemoryRouter><NodeInspectorModal isOpen onClose={onClose} target={{ kind: 'pipelineRun', pipelineId: 'pipe' }} nodeId="train-1" /></MemoryRouter>);
    expect(await screen.findByText('Zero time')).toBeInTheDocument();
    expect(monitoringApi.getPipelineRunNode).toHaveBeenCalledWith('pipe', 'train-1');
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByText('0.00s')).toBeInTheDocument();
    expect(screen.getByText(/unparsed timestamp/)).toBeInTheDocument();
    expect(screen.getByText('No parameters recorded.')).toBeInTheDocument();
    expect(screen.getAllByText(/Repeated/)).toHaveLength(2);
    fireEvent.keyDown(document, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('keeps the newer node details when an earlier response arrives last', async () => {
    // A delayed response must not show one node's details under another node's selection.
    let finishFirst!: (response: NodeInspectorResponse) => void;
    vi.mocked(monitoringApi.getJobNode).mockImplementation((_job, id) => id === 'first'
      ? new Promise(resolve => { finishFirst = resolve; })
      : Promise.resolve(baseResponse({ node: { node_id: 'second', label: 'Second response', step_type: 'training', params: {}, upstream: [], downstream: [] } })));
    const target = { kind: 'job', jobId: 'job-1' } as const;
    const { rerender } = render(<MemoryRouter><NodeInspectorModal isOpen onClose={vi.fn()} target={target} nodeId="first" /></MemoryRouter>);
    rerender(<MemoryRouter><NodeInspectorModal isOpen onClose={vi.fn()} target={target} nodeId="second" /></MemoryRouter>);
    expect(await screen.findByText('Second response')).toBeInTheDocument();
    await act(async () => { finishFirst(baseResponse({ node: { node_id: 'first', label: 'First response', step_type: 'training', params: {}, upstream: [], downstream: [] } })); });
    expect(screen.getByText('Second response')).toBeInTheDocument();
    expect(screen.queryByText('First response')).not.toBeInTheDocument();
  });

  it('keeps loading the selected node when an older request finishes first', async () => {
    // An obsolete finally block must not hide the active request's loading state.
    const first = deferredNode();
    const second = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockImplementation((_job, id) => id === 'first' ? first.promise : second.promise);
    const view = mountInspector();
    view.update({ nodeId: 'second' });
    await act(async () => { first.resolve(nodeResponse('first', 'Old detail')); });
    expect(screen.getByRole('status')).toHaveTextContent('Loading node detail');
    expect(screen.queryByText('Old detail')).not.toBeInTheDocument();
    await act(async () => { second.resolve(nodeResponse('second', 'Selected detail')); });
    expect(screen.getByText('Selected detail')).toBeInTheDocument();
  });

  it.each([false, true])('ignores an older error when the selected response is already loaded: %s', async (loaded) => {
    // Old failures must neither replace current content nor end its loading state early.
    const first = deferredNode();
    const second = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockImplementation((_job, id) => id === 'first' ? first.promise : second.promise);
    const view = mountInspector();
    view.update({ nodeId: 'second' });
    if (loaded) await act(async () => { second.resolve(nodeResponse('second', 'Selected detail')); });
    await act(async () => { first.reject(new Error('Obsolete failure')); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    if (!loaded) {
      expect(screen.getByRole('status')).toHaveTextContent('Loading node detail');
      await act(async () => { second.resolve(nodeResponse('second', 'Selected detail')); });
    }
    expect(screen.getByText('Selected detail')).toBeInTheDocument();
  });

  it('keeps a pipeline-run response when the previous job response arrives later', async () => {
    // Identical node IDs in different run sources must have separate request lifetimes.
    const oldJob = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockReturnValue(oldJob.promise);
    vi.mocked(monitoringApi.getPipelineRunNode).mockResolvedValue(nodeResponse('first', 'Pipeline detail'));
    const view = mountInspector();
    view.update({ target: { kind: 'pipelineRun', pipelineId: 'pipeline-2' } });
    expect(await screen.findByText('Pipeline detail')).toBeInTheDocument();
    await act(async () => { oldJob.resolve(nodeResponse('first', 'Old job detail')); });
    expect(monitoringApi.getPipelineRunNode).toHaveBeenLastCalledWith('pipeline-2', 'first');
    expect(screen.getByText('Pipeline detail')).toBeInTheDocument();
    expect(screen.queryByText('Old job detail')).not.toBeInTheDocument();
  });

  it.each(['success', 'error'] as const)('ignores a previous opening\'s late %s after the same node is reopened', async (outcome) => {
    // Closing invalidates pending work even when the next opening has the same target and node.
    const oldOpening = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockReturnValueOnce(oldOpening.promise)
      .mockResolvedValue(nodeResponse('first', 'Reopened detail'));
    const view = mountInspector();
    view.update({ isOpen: false });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    view.update({ isOpen: true });
    expect(await screen.findByText('Reopened detail')).toBeInTheDocument();
    await act(async () => {
      if (outcome === 'success') oldOpening.resolve(nodeResponse('first', 'Closed detail'));
      else oldOpening.reject(new Error('Closed failure'));
    });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText('Reopened detail')).toBeInTheDocument();
    expect(screen.queryByText('Closed detail')).not.toBeInTheDocument();
  });

  it('ignores a pending retry after navigating to another node', async () => {
    // Retry belongs to the failed selection and must not overwrite a newer selection.
    const retry = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockRejectedValueOnce(new Error('First attempt failed'))
      .mockReturnValueOnce(retry.promise).mockResolvedValue(nodeResponse('second', 'Selected detail'));
    const view = mountInspector();
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }));
    expect(screen.getByRole('status')).toHaveTextContent('Loading node detail');
    view.update({ nodeId: 'second' });
    expect(await screen.findByText('Selected detail')).toBeInTheDocument();
    await act(async () => { retry.reject(new Error('Old retry failed')); });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText('Selected detail')).toBeInTheDocument();
  });

  it('allows the current failed request to be retried successfully', async () => {
    // Stale-response protection must not suppress a legitimate retry's result.
    vi.mocked(monitoringApi.getJobNode).mockRejectedValueOnce(new Error('Temporary failure'))
      .mockResolvedValue(nodeResponse('first', 'Recovered detail'));
    mountInspector();
    fireEvent.click(await screen.findByRole('button', { name: 'Retry' }));
    expect(await screen.findByText('Recovered detail')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('invalidates requests from a discarded effect during Strict Mode replay', async () => {
    // Effect teardown must revoke old writes just as it does on component unmount.
    const discarded = deferredNode();
    vi.mocked(monitoringApi.getJobNode).mockReturnValueOnce(discarded.promise)
      .mockResolvedValue(nodeResponse('first', 'Active detail'));
    render(<StrictMode><MemoryRouter><NodeInspectorModal isOpen onClose={vi.fn()}
      target={{ kind: 'job', jobId: 'job-1' }} nodeId="first" /></MemoryRouter></StrictMode>);
    expect(await screen.findByText('Active detail')).toBeInTheDocument();
    await act(async () => { discarded.resolve(nodeResponse('first', 'Discarded detail')); });
    expect(screen.getByText('Active detail')).toBeInTheDocument();
    expect(screen.queryByText('Discarded detail')).not.toBeInTheDocument();
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
