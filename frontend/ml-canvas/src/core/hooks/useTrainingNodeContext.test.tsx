import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { jobsApi, type RunPipelineResponse } from '../api/jobs';
import { useGraphStore } from '../store/useGraphStore';
import { useJobStore } from '../store/useJobStore';
import { useViewStore } from '../store/useViewStore';
import { warnAndBlockOnLeakage } from '../utils/pipelineLeakageValidation';
import { useTrainingNodeContext } from './useTrainingNodeContext';
import { toast } from '../toast';
import { convertGraphToPipelineConfig } from '../utils/pipelineConverter';
import type { PipelineConfigModel } from '../api/client';

vi.mock('./useUpstreamData', () => ({ useUpstreamData: () => [] }));
vi.mock('./useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: undefined }) }));
vi.mock('../utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: vi.fn(() => ({ nodes: [] })) }));
vi.mock('../utils/pipelineLeakageValidation', () => ({ warnAndBlockOnLeakage: vi.fn(() => false) }));
vi.mock('../api/jobs', () => ({ jobsApi: { runPipeline: vi.fn() } }));
vi.mock('../toast', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const response = { job_id: 'job-a', job_ids: ['job-a'], pipeline_id: 'run', message: 'Submitted' };

beforeEach(() => {
  vi.clearAllMocks();
  useViewStore.setState({ leakageNotice: null, isResultsPanelExpanded: false });
  vi.mocked(warnAndBlockOnLeakage).mockReturnValue(false);
  vi.mocked(convertGraphToPipelineConfig).mockReturnValue({ pipeline_id: 'pipeline', nodes: [] });
  useGraphStore.setState({
    nodes: ['dataset', 'model-a', 'model-b'].map((id, index) => ({
      id, position: { x: index * 200, y: 0 }, data: index === 0
        ? { definitionType: 'dataset_node', datasetId: 'dataset-1' }
        : { definitionType: 'classification', model_type: 'random_forest_classifier' },
    })),
    edges: ['model-a', 'model-b'].map(target => ({ id: target, source: 'dataset', target })),
  });
  useJobStore.setState({ jobs: [], nodeSubmissions: {}, inspectedRun: null, toggleDrawer: vi.fn(), setTab: vi.fn(), startPolling: vi.fn(), setActiveParallelRun: vi.fn() });
});

it('submits training once while pending and follows even a single background job', async () => {
  // Rapid activation must not create duplicate experiments or stop monitoring single jobs.
  let resolve!: (value: RunPipelineResponse) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValue(new Promise(done => { resolve = done; }));
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  let pending!: Promise<void>;
  act(() => {
    pending = result.current.runJob('training', 'classification');
    void result.current.runJob('training', 'classification');
  });
  expect(result.current.isSubmitting).toBe(true);
  expect(jobsApi.runPipeline).toHaveBeenCalledTimes(1);
  await act(async () => { resolve(response); await pending; });
  expect(result.current.isSubmitting).toBe(false);
  expect(result.current.runFeedback).toEqual({ label: 'Training — random forest classifier', jobIds: ['job-a'] });
  expect(useJobStore.getState().startPolling).toHaveBeenCalledOnce();
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('classification');
  expect(useJobStore.getState().inspectedRun).toBeNull();
});

it('leaves the previous Run all scope when submitting a node with multiple jobs', async () => {
  // Node actions must open their normal model history even when the backend fans out.
  useJobStore.setState({ inspectedRun: { label: 'Experiments', jobIds: ['older'] } });
  vi.mocked(jobsApi.runPipeline).mockResolvedValue({ ...response, job_ids: ['job-a', 'job-b'] });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('tuning', 'classification'); });
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('classification');
  expect(useJobStore.getState().inspectedRun).toBeNull();
});

it('keeps an earlier node submission out of the next node settings', async () => {
  // Delayed responses must not put another node's run status in the current editor.
  let resolve!: (value: RunPipelineResponse) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValue(new Promise(done => { resolve = done; }));
  const { result, rerender } = renderHook(({ id }) => useTrainingNodeContext(id), { initialProps: { id: 'model-a' } });
  let pending!: Promise<void>;
  act(() => { pending = result.current.runJob('tuning', 'classification'); });
  rerender({ id: 'model-b' });
  expect(result.current.isSubmitting).toBe(false);
  await act(async () => { resolve(response); await pending; });
  expect(result.current.runFeedback).toBeNull();
  rerender({ id: 'model-a' });
  expect(result.current.runFeedback?.label).toBe('Tuning — random forest classifier');
});

it('retains the pending guard when the keyed settings editor remounts', async () => {
  // Selecting another node or closing settings must not allow a second pending POST.
  let resolve!: (value: RunPipelineResponse) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValue(new Promise(done => { resolve = done; }));
  const first = renderHook(() => useTrainingNodeContext('model-a'));
  let pending!: Promise<void>;
  act(() => { pending = first.result.current.runJob('training', 'classification'); });
  first.unmount();
  const reopened = renderHook(() => useTrainingNodeContext('model-a'));
  expect(reopened.result.current.isSubmitting).toBe(true);
  await act(async () => { await reopened.result.current.runJob('training', 'classification'); });
  await act(async () => { resolve(response); await pending; });
  expect(reopened.result.current.runFeedback?.jobIds).toEqual(['job-a']);
  expect(jobsApi.runPipeline).toHaveBeenCalledTimes(1);
});

it('exposes a visible actionable reason when leakage blocks submission', async () => {
  // Users must be able to correct a blocked action after its toast disappears.
  vi.mocked(warnAndBlockOnLeakage).mockReturnValue(true);
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('training', 'classification'); });
  expect(result.current.submissionMessage).toContain('after the train/test split');
  expect(result.current.isSubmitting).toBe(false);
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** Selecting a safe branch must ignore sibling leakage while unsafe selected ancestors stay blocked. */
it.each([
  { targetNodeId: 'model-a', shouldSubmit: true },
  { targetNodeId: 'model-b', shouldSubmit: false },
])('scopes leakage preflight to selected target $targetNodeId', async ({ targetNodeId, shouldSubmit }) => {
  const leakage = await vi.importActual<typeof import('../utils/pipelineLeakageValidation')>('../utils/pipelineLeakageValidation');
  vi.mocked(warnAndBlockOnLeakage).mockImplementation(leakage.warnAndBlockOnLeakage);
  const splitterParams = { target_column: 'target', test_size: 0.2, random_state: 42, shuffle: true };
  const trainingParams = {
    model_type: 'random_forest_classifier', task_type: 'classification', target_column: 'target',
    hyperparameters: { n_estimators: 10, random_state: 42 }, cv_enabled: false,
  };
  const config: PipelineConfigModel = {
    pipeline_id: 'mixed-leakage-branches',
    nodes: [
      { node_id: 'dataset', step_type: 'data_loader', params: { dataset_id: 'dataset-1' }, inputs: [] },
      { node_id: 'split-safe', step_type: 'TrainTestSplitter', params: splitterParams, inputs: ['dataset'] },
      { node_id: 'scale-safe', step_type: 'StandardScaler', params: { columns: ['feature'] }, inputs: ['split-safe'] },
      { node_id: 'model-a', step_type: 'training', params: trainingParams, inputs: ['scale-safe'] },
      { node_id: 'scale-unsafe', step_type: 'StandardScaler', params: { columns: ['feature'] }, inputs: ['dataset'] },
      { node_id: 'split-unsafe', step_type: 'TrainTestSplitter', params: splitterParams, inputs: ['scale-unsafe'] },
      { node_id: 'model-b', step_type: 'training', params: trainingParams, inputs: ['split-unsafe'] },
    ],
  };
  vi.mocked(convertGraphToPipelineConfig).mockReturnValue(config);
  vi.mocked(jobsApi.runPipeline).mockResolvedValue(response);
  useGraphStore.setState({
    nodes: config.nodes.map((node, index) => ({
      id: node.node_id, position: { x: index * 200, y: 0 },
      data: { ...node.params, definitionType: node.step_type === 'training' ? 'classification' : node.step_type },
    })),
    edges: config.nodes.flatMap(node => node.inputs.map(source => ({
      id: `${source}-${node.node_id}`, source, target: node.node_id,
    }))),
  });
  const { result } = renderHook(() => useTrainingNodeContext(targetNodeId));
  await act(async () => { await result.current.runJob('training', 'classification'); });
  expect(result.current.isSubmitting).toBe(false);
  if (shouldSubmit) {
    expect(result.current.runFeedback?.jobIds).toEqual(['job-a']);
    expect(jobsApi.runPipeline).toHaveBeenCalledExactlyOnceWith({
      ...config, target_node_id: targetNodeId, job_type: 'training',
    });
  } else {
    expect(result.current.submissionMessage).toContain('after the train/test split');
    expect(jobsApi.runPipeline).not.toHaveBeenCalled();
    expect(toast.error).toHaveBeenCalledWith('Data leakage risk detected', expect.stringContaining("'scale-unsafe'"));
  }
});

/** Server-side leakage failures must explain how to correct the graph in persistent feedback. */
it.each(['training', 'tuning'] as const)('shows the HTTP 400 leakage detail for %s submissions', async (jobType) => {
  const detail = "Data leakage risk: node 'scale' (StandardScaler) fits on unsplit data. Move it after the train/test splitter.";
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(Object.assign(new Error(detail), {
    response: { status: 400, data: { detail } },
  }));
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob(jobType, 'classification'); });
  expect(result.current.submissionMessage).toContain(detail);
  expect(result.current.isSubmitting).toBe(false);
  expect(result.current.runFeedback).toBeNull();
  expect(useJobStore.getState().startPolling).not.toHaveBeenCalled();
  expect(toast.error).toHaveBeenCalledWith('Failed to submit job', detail);
  expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
  expect(useViewStore.getState().isResultsPanelExpanded).toBe(false);
});

/** Retrying a model must remove stale safety feedback even while its new request is pending. */
it('clears safety notices on a fresh training request and preserves structured backend detail', async () => {
  useViewStore.setState({ leakageNotice: { message: 'Earlier failure', graphSignature: 'old' } });
  const detail = 'Per-fold preprocessing refit skipped: unsupported graph; CV/tuning scores may be optimistically biased.';
  let reject!: (reason: unknown) => void;
  vi.mocked(jobsApi.runPipeline).mockReturnValueOnce(new Promise((_, fail) => { reject = fail; }));
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  let pending!: Promise<void>;
  act(() => { pending = result.current.runJob('training', 'classification'); });
  expect(useViewStore.getState().leakageNotice).toBeNull();
  act(() => useGraphStore.setState({ nodes: [], edges: [] }));
  await act(async () => { reject({ response: { data: { detail } } }); await pending; });
  expect(useViewStore.getState().leakageNotice?.message).toBe(detail);
  expect(useViewStore.getState().leakageNotice?.graphSignature).toContain('model-a');
  expect(useJobStore.getState().nodeSubmissions['model-a']?.message).toContain(detail);
});

/** A rejection without an error message must still leave actionable feedback. */
it('shows fallback feedback when a submission rejects without a message', async () => {
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(new Error());
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('training', 'classification'); });
  expect(result.current.submissionMessage).toContain('Check your connection and settings, then try again.');
  expect(result.current.isSubmitting).toBe(false);
});

it('clears the pending guard after rejection and identifies a later tuning submission', async () => {
  // A failed request must remain actionable and must not strand the Train/Tune button.
  vi.mocked(jobsApi.runPipeline).mockRejectedValueOnce(new Error('Offline')).mockResolvedValueOnce(response);
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('tuning', 'classification'); });
  expect(result.current.isSubmitting).toBe(false);
  expect(result.current.submissionMessage).toContain('Tuning — random forest classifier: Submission failed');
  await act(async () => { await result.current.runJob('tuning', 'classification'); });
  expect(result.current.submissionMessage).toBe('');
  expect(result.current.runFeedback?.jobIds).toEqual(['job-a']);
  expect(jobsApi.runPipeline).toHaveBeenCalledTimes(2);
});

/** Missing editor context must not accidentally submit every model in the graph. */
it.each([undefined, 'deleted-model'])('does not submit when node %s is unavailable', async (nodeId) => {
  const { result } = renderHook(() => useTrainingNodeContext(nodeId));
  await act(async () => { await result.current.runJob('training', 'classification'); });
  expect(result.current.datasetId).toBeUndefined();
  expect(result.current.isSubmitting).toBe(false);
  expect(result.current.runFeedback).toBeNull();
  expect(useJobStore.getState().nodeSubmissions).toEqual({});
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** A disconnected model needs persistent instructions even when there is no toast. */
it.each([
  { data: { label: 'Custom_classifier' }, modelName: 'Custom classifier' },
  { data: {}, modelName: 'selected model' },
])('identifies a disconnected $modelName in blocked feedback', async ({ data, modelName }) => {
  useGraphStore.setState({ nodes: [{ id: 'model-a', position: { x: 0, y: 0 }, data }], edges: [] });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('tuning', 'classification'); });
  expect(result.current.submissionMessage).toContain(modelName);
  expect(result.current.submissionMessage).toContain('Connect a dataset upstream and select a dataset');
  expect(result.current.isSubmitting).toBe(false);
  expect(result.current.runFeedback).toBeNull();
  expect(jobsApi.runPipeline).not.toHaveBeenCalled();
});

/** Legacy responses without a nonempty batch list must still be monitored. */
it('falls back to job_id when the response has no batch jobs', async () => {
  vi.mocked(jobsApi.runPipeline).mockResolvedValue({ ...response, job_ids: [] });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  await act(async () => { await result.current.runJob('training', 'classification'); });
  expect(result.current.runFeedback?.jobIds).toEqual(['job-a']);
  expect(useJobStore.getState().startPolling).toHaveBeenCalledOnce();
  expect(useJobStore.getState().setActiveParallelRun).not.toHaveBeenCalled();
});

/** Dataset resolution must preserve legacy field precedence without using the selected model's data. */
it.each([
  { data: { datasetId: 'direct', dataset_id: 'legacy', config: { datasetId: 'config' } }, expected: 'direct' },
  { data: { datasetId: '', dataset_id: 'legacy', config: { dataset_id: 'config' } }, expected: 'config' },
  { data: { config: { datasetId: '', dataset_id: 'legacy' }, params: { dataset_id: 'params' } }, expected: 'params' },
])('resolves upstream dataset $expected with nullish field precedence', ({ data, expected }) => {
  useGraphStore.setState({ nodes: [
    { id: 'dataset', position: { x: 0, y: 0 }, data },
    { id: 'model-a', position: { x: 1, y: 0 }, data: { datasetId: 'ignore-self' } },
  ], edges: [{ id: 'edge', source: 'dataset', target: 'model-a' }] });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  expect(result.current.datasetId).toBe(expected);
});

/** Breadth-first graph order must win over edge order and terminate even with an upstream cycle. */
it('resolves the nearest dataset in node order across a cyclic graph', () => {
  useGraphStore.setState({ nodes: [
    { id: 'model-a', position: { x: 0, y: 0 }, data: {} },
    { id: 'first', position: { x: 0, y: 0 }, data: { dataset_id: 'first-dataset' } },
    { id: 'second', position: { x: 0, y: 0 }, data: { dataset_id: 'second-dataset' } },
    { id: 'middle', position: { x: 0, y: 0 }, data: {} },
  ], edges: [
    { id: '2', source: 'second', target: 'middle' }, { id: '1', source: 'first', target: 'middle' },
    { id: 'm', source: 'middle', target: 'model-a' }, { id: 'cycle', source: 'model-a', target: 'middle' },
  ] });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  expect(result.current.datasetId).toBe('first-dataset');
});

/** Submission feedback must be committed before polling and opening the chosen task history. */
it('preserves parallel tuning payloads and store action order', async () => {
  const order: string[] = [];
  const setNodeSubmission = useJobStore.getState().setNodeSubmission;
  const setInspectedRun = useJobStore.getState().setInspectedRun;
  const cfg: PipelineConfigModel = { pipeline_id: 'full-graph', nodes: [
    { node_id: 'model-a', step_type: 'training', inputs: [], params: { target_column: 'target' } },
    { node_id: 'sibling', step_type: 'training', inputs: [], params: {} },
  ] };
  vi.mocked(convertGraphToPipelineConfig).mockReturnValue(cfg);
  vi.mocked(jobsApi.runPipeline).mockResolvedValue({ ...response, job_ids: ['job-b', 'job-a'] });
  useJobStore.setState({
    setNodeSubmission: (id, value) => { order.push(value.pending ? 'pending' : 'accepted'); setNodeSubmission(id, value); },
    startPolling: vi.fn(() => { order.push('poll'); }),
    setActiveParallelRun: vi.fn(() => { order.push('parallel'); }),
    setTab: vi.fn(() => { order.push('tab'); }),
    setInspectedRun: (run) => { order.push('inspection'); setInspectedRun(run); },
    toggleDrawer: vi.fn(() => { order.push('drawer'); }),
  });
  const { result } = renderHook(() => useTrainingNodeContext('model-a'));
  try {
    await act(async () => { await result.current.runJob('tuning', 'regression'); });
    expect(jobsApi.runPipeline).toHaveBeenCalledExactlyOnceWith({ ...cfg, target_node_id: 'model-a', job_type: 'tuning' });
    expect(warnAndBlockOnLeakage).toHaveBeenCalledExactlyOnceWith({ nodes: [cfg.nodes[0]] });
    expect(result.current.runFeedback).toEqual({ label: 'Tuning — random forest classifier', jobIds: ['job-b', 'job-a'] });
    expect(order).toEqual(['pending', 'accepted', 'poll', 'parallel', 'tab', 'inspection', 'drawer']);
    expect(useJobStore.getState().setTab).toHaveBeenCalledWith('regression');
    expect(useJobStore.getState().setActiveParallelRun).toHaveBeenCalledWith({ jobIds: ['job-b', 'job-a'], startedAt: expect.any(String) });
    expect(toast.success).toHaveBeenCalledWith('Parallel execution started', '2 branches submitted.');
  } finally {
    useJobStore.setState({ setNodeSubmission, setInspectedRun });
  }
});
