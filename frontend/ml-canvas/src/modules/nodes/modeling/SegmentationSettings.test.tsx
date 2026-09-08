import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { jobsApi, type RunPipelineResponse } from '../../../core/api/jobs';
import { registryApi } from '../../../core/api/registry';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { warnAndBlockOnLeakage } from '../../../core/utils/pipelineLeakageValidation';
import { SegmentationSettings, type SegmentationConfig } from './SegmentationSettings';

vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: undefined }) }));
vi.mock('../../../core/api/registry', () => ({ registryApi: { getAllNodes: vi.fn() } }));
vi.mock('../../../core/api/jobs', () => ({ jobsApi: { getHyperparameters: vi.fn(), runPipeline: vi.fn() } }));
vi.mock('../../../core/utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: () => ({ nodes: [] }) }));
vi.mock('../../../core/utils/pipelineLeakageValidation', () => ({ warnAndBlockOnLeakage: vi.fn(() => false) }));
vi.mock('../../../core/toast', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const config: SegmentationConfig = { model_type: 'kmeans', hyperparameters: { n_clusters: 3 } };
const response: RunPipelineResponse = { job_id: 'job-a', job_ids: ['job-a'], pipeline_id: 'pipeline-1', message: 'Submitted' };

async function renderSettings(patch: Partial<SegmentationConfig> = {}, nodeId: string | undefined = 'segment-a') {
  let view!: ReturnType<typeof render>;
  await act(async () => {
    view = render(<SegmentationSettings config={{ ...config, ...patch }} onChange={vi.fn()} {...(nodeId ? { nodeId } : {})} />);
  });
  return view;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('IntersectionObserver', undefined);
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([
    { id: 'kmeans', name: 'K-Means', category: 'Modeling', description: '', params: {}, tags: ['clustering'] },
  ]);
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([]);
  vi.mocked(jobsApi.runPipeline).mockResolvedValue(response);
  vi.mocked(warnAndBlockOnLeakage).mockReturnValue(false);
  useGraphStore.setState({
    nodes: ['dataset', 'segment-a', 'segment-b'].map((id, index) => ({
      id, position: { x: index * 100, y: 0 }, data: index === 0
        ? { definitionType: 'dataset_node', datasetId: 'dataset-1' }
        : { definitionType: 'segmentation', model_type: 'kmeans' },
    })),
    edges: ['segment-a', 'segment-b'].map(target => ({ id: target, source: 'dataset', target })),
  });
  useJobStore.setState({
    jobs: [], runJobs: {}, nodeSubmissions: {}, inspectedRun: { label: 'Previous pipeline', jobIds: ['old-job'] },
    toggleDrawer: vi.fn(), setTab: vi.fn(), startPolling: vi.fn(), setActiveParallelRun: vi.fn(),
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Clustering's fixed and dynamic fields must expose the captions users see. */
it('labels clustering configuration and dynamic parameters', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([
    { name: 'n_clusters', label: 'Clusters', type: 'number', default: 3 },
    { name: 'init', label: 'Initialization', type: 'select', default: 'random', options: [{ label: 'Random', value: 'random' }] },
  ]);
  await renderSettings();
  expect(screen.getByRole('combobox', { name: 'Clustering Algorithm' })).toBeVisible();
  expect(screen.getByRole('combobox', { name: 'Reference Column (optional)' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByRole('textbox', { name: 'Clusters' })).toHaveValue('3');
  expect(screen.getByRole('combobox', { name: 'Initialization' })).toBeVisible();
});

/** Unsupervised training needs a dataset and algorithm, but no target column. */
it.each([
  { name: 'missing dataset', connected: false, model_type: 'kmeans', reason: /Connect a dataset node upstream/ },
  { name: 'missing algorithm', connected: true, model_type: '', reason: /Choose a clustering algorithm/ },
])('blocks segmentation with $name', async ({ connected, model_type, reason }) => {
  if (!connected) useGraphStore.setState({ edges: [] });
  await renderSettings({ model_type });
  const action = screen.getByRole('button', { name: 'Train segmentation' });
  expect(action).toBeDisabled();
  expect(action).toHaveAccessibleDescription(reason);
  fireEvent.click(action);
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** An editor without a node must stay harmless rather than submitting a whole graph. */
it('blocks settings without a node id', async () => {
  await renderSettings({}, '');
  const action = screen.getByRole('button', { name: 'Train segmentation' });
  fireEvent.click(action);
  expect(action).toBeDisabled();
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** Both backend response shapes must create a receipt and start monitoring every job. */
it.each([
  { name: 'single job', jobIds: ['job-a'], expectedIds: ['job-a'] },
  { name: 'legacy empty job list', jobIds: [], expectedIds: ['job-a'] },
  { name: 'parallel jobs', jobIds: ['job-a', 'job-b'], expectedIds: ['job-a', 'job-b'] },
])('follows $name and opens segmentation history', async ({ jobIds, expectedIds }) => {
  vi.mocked(jobsApi.runPipeline).mockResolvedValue({ ...response, job_ids: jobIds });
  await renderSettings();
  const action = screen.getByRole('button', { name: 'Train segmentation' });
  expect(action).toBeEnabled();
  expect(action).toHaveAccessibleDescription(/Trains K-Means in the background without a target column/);
  await act(async () => fireEvent.click(action));
  expect(jobsApi.runPipeline).toHaveBeenCalledExactlyOnceWith({ nodes: [], target_node_id: 'segment-a', job_type: 'training' });
  expect(useJobStore.getState().nodeSubmissions['segment-a']).toMatchObject({
    pending: false, message: '', run: { jobIds: expectedIds },
  });
  expect(screen.getByRole('status')).toHaveTextContent(/Segmentation.*kmeans: Awaiting status/);
  expect(useJobStore.getState().startPolling).toHaveBeenCalledOnce();
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('segmentation');
  expect(useJobStore.getState().inspectedRun).toBeNull();
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
  if (expectedIds.length > 1) {
    expect(useJobStore.getState().setActiveParallelRun).toHaveBeenCalledWith({ jobIds: expectedIds, startedAt: expect.any(String) });
  } else {
    expect(useJobStore.getState().setActiveParallelRun).not.toHaveBeenCalled();
  }
});

/** Closing and reopening settings must preserve the guard against duplicate POSTs. */
it('keeps a pending submission disabled across remounts', async () => {
  let resolve!: (value: RunPipelineResponse) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValue(new Promise(done => { resolve = done; }));
  const first = await renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' }));
  expect(screen.getByRole('button', { name: 'Submitting job...' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent(/Submitting/);
  first.unmount();
  await renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Submitting job...' }));
  expect(jobsApi.runPipeline).toHaveBeenCalledTimes(1);
  await act(async () => resolve(response));
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
  expect(screen.getByRole('button', { name: 'View jobs' })).toBeVisible();
});

/** Leakage prevention must leave persistent corrective feedback and release the pending lock. */
it('explains a blocked preprocessing graph without posting a job', async () => {
  vi.mocked(warnAndBlockOnLeakage).mockReturnValue(true);
  await renderSettings();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' })));
  expect(screen.getByRole('status')).toHaveTextContent(/Move data-learning preprocessing after the train\/test split/);
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
  expect(useJobStore.getState().nodeSubmissions['segment-a']?.pending).toBe(false);
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** A failed POST must be retryable and replaced by the next successful receipt. */
it('recovers from a failed submission', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(new Error('Offline')).mockResolvedValueOnce(response);
  await renderSettings();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' })));
  expect(screen.getByRole('status')).toHaveTextContent(/Submission failed.*try again/);
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' })));
  expect(jobsApi.runPipeline).toHaveBeenCalledTimes(2);
  expect(screen.getByRole('status')).toHaveTextContent(/Awaiting status/);
  expect(screen.queryByText(/Submission failed/)).not.toBeInTheDocument();
});

/** Delayed responses must remain attached to their original node after selection changes. */
it('keeps feedback scoped to the submitted segmentation node', async () => {
  let resolve!: (value: RunPipelineResponse) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValue(new Promise(done => { resolve = done; }));
  const view = await renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' }));
  view.rerender(<SegmentationSettings config={config} onChange={vi.fn()} nodeId="segment-b" />);
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
  await act(async () => resolve(response));
  expect(screen.queryByRole('status')).not.toBeInTheDocument();
  view.rerender(<SegmentationSettings config={config} onChange={vi.fn()} nodeId="segment-a" />);
  expect(screen.getByRole('status')).toHaveTextContent(/Awaiting status/);
  fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
  expect(useJobStore.getState().setTab).toHaveBeenLastCalledWith('segmentation');
});

/** Legacy dataset locations must remain usable through preprocessing and cyclic graph edges. */
it.each([
  { datasetId: 'dataset-1' }, { dataset_id: 'dataset-1' },
  { config: { datasetId: 'dataset-1' } }, { config: { dataset_id: 'dataset-1' } },
  { params: { datasetId: 'dataset-1' } }, { params: { dataset_id: 'dataset-1' } },
])('resolves an upstream dataset stored as %j', async (data) => {
  useGraphStore.setState({
    nodes: [
      { id: 'dataset', position: { x: 0, y: 0 }, data },
      { id: 'transform', position: { x: 100, y: 0 }, data: {} },
      { id: 'segment-a', position: { x: 200, y: 0 }, data: { model_type: 'kmeans' } },
    ],
    edges: [
      { id: 'data', source: 'dataset', target: 'transform' },
      { id: 'model', source: 'transform', target: 'segment-a' },
      { id: 'cycle', source: 'segment-a', target: 'transform' },
    ],
  });
  await renderSettings({ model_type: 'custom_clustering' });
  const action = screen.getByRole('button', { name: 'Train segmentation' });
  expect(action).toBeEnabled();
  expect(action).toHaveAccessibleDescription(/Trains custom clustering in the background/);
});
