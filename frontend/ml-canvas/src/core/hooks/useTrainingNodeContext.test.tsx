import { act, renderHook } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { jobsApi, type RunPipelineResponse } from '../api/jobs';
import { useGraphStore } from '../store/useGraphStore';
import { useJobStore } from '../store/useJobStore';
import { warnAndBlockOnLeakage } from '../utils/pipelineLeakageValidation';
import { useTrainingNodeContext } from './useTrainingNodeContext';

vi.mock('./useUpstreamData', () => ({ useUpstreamData: () => [] }));
vi.mock('./useDatasetSchema', () => ({ useDatasetSchema: () => ({ data: undefined }) }));
vi.mock('../utils/pipelineConverter', () => ({ convertGraphToPipelineConfig: () => ({ nodes: [] }) }));
vi.mock('../utils/pipelineLeakageValidation', () => ({ warnAndBlockOnLeakage: vi.fn(() => false) }));
vi.mock('../api/jobs', () => ({ jobsApi: { runPipeline: vi.fn() } }));
vi.mock('../toast', () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const response = { job_id: 'job-a', job_ids: ['job-a'], pipeline_id: 'run', message: 'Submitted' };

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(warnAndBlockOnLeakage).mockReturnValue(false);
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
