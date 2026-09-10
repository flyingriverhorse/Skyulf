import { useState } from 'react';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { jobsApi, type RunPipelineResponse } from '../../../core/api/jobs';
import { registryApi } from '../../../core/api/registry';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { useViewStore } from '../../../core/store/useViewStore';
import { warnAndBlockOnLeakage } from '../../../core/utils/pipelineLeakageValidation';
import { SegmentationSettings, type SegmentationConfig } from './SegmentationSettings';
import { ValidationNavigation } from '../../../components/shared/ValidationField';
import type { HyperparameterDef } from './components/types';

const presentation = vi.hoisted(() => ({ wide: false, schema: vi.fn() }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, presentation.wide] }));
vi.mock('../../../core/hooks/useDatasetSchema', () => ({ useDatasetSchema: presentation.schema }));
vi.mock('../../../core/api/registry', () => ({ registryApi: { getAllNodes: vi.fn() } }));
vi.mock('../../../core/api/jobs', () => ({ jobsApi: { getHyperparameters: vi.fn(), runPipeline: vi.fn() } }));
vi.mock('../../../core/utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: () => ({ nodes: [] }) }));
vi.mock('../../../core/utils/pipelineLeakageValidation', () => ({
  warnAndBlockOnLeakage: vi.fn(() => false), findPreprocessingBeforeSplitIssues: () => [],
}));
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
  sessionStorage.clear();
  presentation.wide = false;
  presentation.schema.mockReturnValue({ data: undefined });
  useViewStore.setState({ leakageNotice: null, validationFocusRequest: null });
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

/** Segmentation safety errors must retain backend corrections in both persistent feedback surfaces. */
it('routes structured backend safety errors to the canvas notice', async () => {
  const detail = 'Data leakage risk: learned preprocessing requires a safe split.';
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce({ response: { data: { detail } } });
  await renderSettings();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Train segmentation' })));
  expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
  expect(screen.getByRole('status')).toHaveTextContent(detail);
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
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

const definitions: HyperparameterDef[] = [
  { name: 'n_clusters', label: 'Clusters', type: 'number', default: 4, min: 2, max: 20, step: 1 },
  { name: 'init', label: 'Initialization', type: 'select', default: 'random', options: [
    { label: 'Random', value: 'random' }, { label: 'Plus plus', value: 'k-means++' },
  ] },
  { name: 'copy_x', label: 'Copy data', type: 'boolean', default: true },
];

/** Drive the same controlled config updates as the public settings inspector. */
function ControlledSettings({ initial = config }: { initial?: SegmentationConfig }) {
  const [value, setValue] = useState(initial);
  return <><SegmentationSettings config={value} onChange={setValue} nodeId="segment-a" />
    <output aria-label="Current config">{JSON.stringify(value)}</output></>;
}

/** Registry categories and clustering tags jointly control the algorithm choices. */
it('filters the registry in server order and keeps scaling dismissal across tab switches', async () => {
  const item = { description: '', params: {} };
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([
    { ...item, id: 'kmeans', name: 'K-Means', category: 'Modeling', tags: ['clustering', 'requires_scaling'] },
    { ...item, id: 'dbscan', name: 'DBSCAN', category: 'Model', tags: ['clustering'] },
    { ...item, id: 'forest', name: 'Forest', category: 'Modeling', tags: ['classification'] },
    { ...item, id: 'prep', name: 'Prep', category: 'Preprocessing', tags: ['clustering'] },
    { ...item, id: 'untagged', name: 'Untagged', category: 'Model' },
  ]);
  await renderSettings();
  expect(within(screen.getByLabelText('Clustering Algorithm')).getAllByRole('option').map(option => option.textContent))
    .toEqual(['K-Means', 'DBSCAN']);
  fireEvent.click(screen.getByRole('button', { name: 'Scale Your Data' }));
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
  expect(screen.getByRole('button', { name: 'Scale Your Data' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.queryByText(/This model performs best/)).not.toBeInTheDocument();
});

/** An unavailable registry retains the historical K-Means fallback. */
it('disables algorithms while loading and falls back after a registry error', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
  let reject!: (error: Error) => void;
  vi.mocked(registryApi.getAllNodes).mockReturnValue(new Promise((_, fail) => { reject = fail; }));
  await renderSettings();
  expect(screen.getByLabelText('Clustering Algorithm')).toBeDisabled();
  await act(async () => reject(new Error('registry offline')));
  expect(screen.getByLabelText('Clustering Algorithm')).toBeEnabled();
  expect(within(screen.getByLabelText('Clustering Algorithm')).getAllByRole('option').map(option => option.textContent))
    .toEqual(['K-Means']);
});

/** Defaults are seeded once and every editor preserves the remaining config fields. */
it('seeds and edits number, select, boolean and reference values without losing execution mode', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue(definitions);
  presentation.schema.mockReturnValue({ data: { columns: { label: { name: 'species' } } } });
  await act(async () => render(<ControlledSettings initial={{ model_type: 'kmeans', hyperparameters: {}, execution_mode: 'parallel' }} />));
  expect(presentation.schema).toHaveBeenLastCalledWith('dataset-1');
  fireEvent.change(screen.getByLabelText('Reference Column (optional)'), { target: { value: 'species' } });
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  const clusters = screen.getByRole('textbox', { name: 'Clusters' });
  expect(clusters).toHaveValue('4');
  expect(clusters).toHaveAttribute('min', '2');
  expect(clusters).toHaveAttribute('max', '20');
  expect(clusters).toHaveAttribute('step', '1');
  fireEvent.change(clusters, { target: { value: '7' } });
  fireEvent.blur(clusters);
  fireEvent.change(screen.getByLabelText('Initialization'), { target: { value: 'k-means++' } });
  fireEvent.change(screen.getByLabelText('Copy data'), { target: { value: 'false' } });
  fireEvent.blur(screen.getByLabelText('Copy data'));
  expect(JSON.parse(screen.getByLabelText('Current config').textContent!)).toEqual({
    model_type: 'kmeans', hyperparameters: { n_clusters: 7, init: 'k-means++', copy_x: false },
    reference_column: 'species', execution_mode: 'parallel',
  });
  fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
  fireEvent.change(screen.getByLabelText('Reference Column (optional)'), { target: { value: '' } });
  expect(JSON.parse(screen.getByLabelText('Current config').textContent!)).not.toHaveProperty('reference_column');
});

/** Changing algorithms clears existing customizations before loading that model's defaults. */
it('clears parameters on model change and seeds the new definitions', async () => {
  const base = { category: 'Modeling', description: '', params: {}, tags: ['clustering'] };
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([
    { ...base, id: 'kmeans', name: 'K-Means' }, { ...base, id: 'dbscan', name: 'DBSCAN' },
  ]);
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValueOnce(definitions);
  let resolve!: (value: HyperparameterDef[]) => void;
  await act(async () => render(<ControlledSettings />));
  vi.mocked(jobsApi.getHyperparameters).mockReturnValueOnce(new Promise(done => { resolve = done; }));
  fireEvent.change(screen.getByLabelText('Clustering Algorithm'), { target: { value: 'dbscan' } });
  expect(JSON.parse(screen.getByLabelText('Current config').textContent!)).toEqual({ model_type: 'dbscan', hyperparameters: {} });
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.queryByLabelText('Clusters')).not.toBeInTheDocument();
  await act(async () => resolve([{ name: 'eps', label: 'Epsilon', type: 'number', default: 0.5 }]));
  expect(screen.getByLabelText('Epsilon')).toHaveValue('0.5');
  expect(JSON.parse(screen.getByLabelText('Current config').textContent!)).toEqual({ model_type: 'dbscan', hyperparameters: { eps: 0.5 } });
});

/** Existing false and zero values survive definition loading and resize remounts. */
it('preserves values while discarding uncommitted drafts on tab and width changes', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue(definitions);
  const initial = { ...config, hyperparameters: { n_clusters: 0, copy_x: false } };
  let view!: ReturnType<typeof render>;
  await act(async () => { view = render(<ControlledSettings initial={initial} />); });
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByLabelText('Clusters')).toHaveValue('0');
  expect(screen.getByLabelText('Copy data')).toHaveValue('false');
  fireEvent.change(screen.getByLabelText('Clusters'), { target: { value: '99' } });
  fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByLabelText('Clusters')).toHaveValue('0');
  fireEvent.change(screen.getByLabelText('Clusters'), { target: { value: '88' } });
  presentation.wide = true;
  view.rerender(<ControlledSettings initial={initial} />);
  expect(screen.queryByRole('button', { name: 'Configuration' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Clustering Algorithm')).toBeVisible();
  expect(screen.getByLabelText('Clusters')).toHaveValue('0');
  presentation.wide = false;
  view.rerender(<ControlledSettings initial={initial} />);
  expect(screen.getByRole('button', { name: 'Hyperparameters' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByLabelText('Clustering Algorithm')).not.toBeInTheDocument();
  expect(jobsApi.getHyperparameters).toHaveBeenCalledTimes(1);
});

/** Pending definitions intentionally use the callback and config captured by the model effect. */
it('keeps the original callback and config when definitions resolve after same-model changes', async () => {
  let resolve!: (value: HyperparameterDef[]) => void;
  vi.mocked(jobsApi.getHyperparameters).mockReturnValue(new Promise(done => { resolve = done; }));
  const original = { ...config, hyperparameters: {}, reference_column: 'old', execution_mode: 'parallel' as const };
  const firstChange = vi.fn();
  const nextChange = vi.fn();
  const view = render(<SegmentationSettings config={original} onChange={firstChange} nodeId="segment-a" />);
  await act(async () => undefined);
  view.rerender(<SegmentationSettings config={{ ...original, reference_column: 'new', execution_mode: 'merge' }} onChange={nextChange} nodeId="segment-a" />);
  await act(async () => resolve(definitions));
  expect(firstChange).toHaveBeenCalledExactlyOnceWith({ ...original, hyperparameters: { n_clusters: 4, init: 'random', copy_x: true } });
  expect(nextChange).not.toHaveBeenCalled();
  expect(jobsApi.getHyperparameters).toHaveBeenCalledTimes(1);
});

/** The original request lifetime permits stale responses after model changes and unmounts. */
it('retains out-of-order definitions and completion after unmount', async () => {
  const resolves: ((value: HyperparameterDef[]) => void)[] = [];
  vi.mocked(jobsApi.getHyperparameters).mockImplementation(() => new Promise(done => { resolves.push(done); }));
  const onChange = vi.fn();
  const view = render(<SegmentationSettings config={{ ...config, hyperparameters: {} }} onChange={onChange} nodeId="segment-a" />);
  await act(async () => undefined);
  view.rerender(<SegmentationSettings config={{ model_type: 'dbscan', hyperparameters: {} }} onChange={onChange} nodeId="segment-a" />);
  await act(async () => resolves[1]!([{ name: 'eps', label: 'Epsilon', type: 'number', default: 0.5 }]));
  expect(onChange).toHaveBeenLastCalledWith({ model_type: 'dbscan', hyperparameters: { eps: 0.5 } });
  view.unmount();
  await act(async () => resolves[0]!(definitions));
  expect(onChange).toHaveBeenLastCalledWith({ model_type: 'kmeans', hyperparameters: { n_clusters: 4, init: 'random', copy_x: true } });
});

/** A rejected definitions request releases the spinner without replacing configured values. */
it('shows empty parameters after a definitions error', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => undefined);
  vi.mocked(jobsApi.getHyperparameters).mockRejectedValue(new Error('definitions offline'));
  await renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByText('No parameters available.')).toBeVisible();
  expect(screen.getByRole('button', { name: 'Train segmentation' })).toBeEnabled();
});

/** The informational dismissal is session-wide while local section state resets on remount. */
it('remembers information dismissal for the session', async () => {
  const view = await renderSettings();
  fireEvent.click(screen.getByRole('button', { name: 'Dismiss segmentation information' }));
  expect(sessionStorage.getItem('hide_info_segmentation')).toBe('true');
  view.unmount();
  await renderSettings();
  expect(screen.queryByText(/Group rows into clusters/)).not.toBeInTheDocument();
});

/** Activating a model validation issue reveals and focuses the algorithm from the params tab. */
it('reveals the model section for validation navigation', async () => {
  await act(async () => render(<ValidationNavigation nodeId="segment-a">
    <SegmentationSettings config={config} onChange={vi.fn()} nodeId="segment-a" />
  </ValidationNavigation>));
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  await act(async () => useViewStore.setState({ validationFocusRequest: {
    nodeId: 'segment-a', nodeLabel: 'Segmentation', field: 'model_type', category: 'configuration', message: 'Choose a model', requestId: 1,
  } }));
  expect(screen.getByRole('button', { name: 'Configuration' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByLabelText('Clustering Algorithm')).toHaveFocus();
});

/** Traversal skips the current node's dataset in favour of its upstream dataset. */
it('ignores dataset metadata on the current node', async () => {
  const graph = useGraphStore.getState();
  useGraphStore.setState({ nodes: graph.nodes.map(node => node.id === 'segment-a'
    ? { ...node, data: { ...node.data, datasetId: 'wrong-self-dataset' } } : node) });
  await renderSettings();
  expect(presentation.schema).toHaveBeenLastCalledWith('dataset-1');
});
