import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { deploymentApi, type PredictionResponse } from '../../../core/api/deployment';
import { jobsApi } from '../../../core/api/jobs';
import { thresholdTuningApi } from '../../../core/api/thresholdTuning';
import { ConfirmProvider } from '../../shared/ConfirmDialog';
import { LS_INPUT, LS_PENDING_RUN, LS_RUN_HISTORY } from './inferenceData';
import { useInferenceController } from './useInferenceController';
import { useInferenceRuns } from './useInferenceRuns';

/** Read the real exported Blob because CSV provenance is the user-visible contract. */
function readBlob(blob: Blob): Promise<string> {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.readAsText(blob);
  });
}

/** Resolve real hook requests without depending on transport timing. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}

describe('inference result input snapshots', () => {
  let exported: Blob;

  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(LS_INPUT, '[{"a":1},{"a":2}]');
    vi.spyOn(deploymentApi, 'getActive').mockResolvedValue({
      id: 1, job_id: 'job-a', model_type: 'regressor', artifact_uri: 'model.joblib',
      is_active: true, created_at: '2026-09-12T00:00:00Z', input_schema: [{ name: 'a', type: 'float64' }],
    });
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue({
      job_id: 'job-a', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
      status: 'completed', start_time: null, end_time: null, error: null, result: null,
      created_at: '2026-09-12T00:00:00Z',
    });
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue({
      thresholds: null, classes: null, metric: null, split_used: null,
      computed_at: null, source: null, enabled: false,
    });
    vi.stubGlobal('URL', Object.assign(URL, {
      createObjectURL: (blob: Blob) => { exported = blob; return 'blob:predictions'; },
      revokeObjectURL: () => {},
    }));
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it('exports and displays the submitted rows after the editor is changed', async () => {
    // Editing or invalidating JSON must not relabel an existing result's rows.
    vi.spyOn(deploymentApi, 'predict').mockResolvedValue({ predictions: [10, 20], model_version: 'v1' });
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment?.job_id).toBe('job-a'));
    await act(async () => { await result.current.handlePredict(); });
    act(() => { result.current.setInputData('[{"a":9},{"a":8},{"a":7}]'); });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n1,10\n2,20');
    expect(result.current.parsedInputRows).toEqual([{ a: 1 }, { a: 2 }]);
    act(() => { result.current.setInputData('{broken'); });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n1,10\n2,20');
  });

  it('keeps the displayed result snapshot until a rerun succeeds', async () => {
    // A pending rerun has its own inputs and cannot change the previous CSV.
    const next = deferred<PredictionResponse>();
    vi.spyOn(deploymentApi, 'predict').mockResolvedValueOnce({ predictions: [10, 20], model_version: 'v1' })
      .mockReturnValueOnce(next.promise);
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment).not.toBeNull());
    await act(async () => { await result.current.handlePredict(); });
    act(() => { result.current.setInputData('[{"a":9}]'); });
    let pending!: Promise<void>;
    act(() => { pending = result.current.handlePredict(); });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n1,10\n2,20');
    act(() => { result.current.setInputData('[{"a":99}]'); });
    await act(async () => { next.resolve({ predictions: [90], model_version: 'v2' }); await pending; });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n9,90');
  });

  it('hydrates saved predictions with their saved inputs while preserving editor text', async () => {
    // Reloading must not join persisted predictions to the separately saved editor.
    localStorage.setItem(LS_RUN_HISTORY, JSON.stringify([{
      runId: 'saved-run', label: 'Run #1', status: 'success', at: Date.now(), rows: 2,
      latencyMs: 1, jobId: 'job-a', modelType: 'regressor', modelVersion: 'v1',
      thresholdContext: 'none', input: '[{"a":3},{"a":4}]', overrideThresholdsUsed: null,
      predictions: [30, 40], errorMessage: null,
    }]));
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.predictions).toEqual([30, 40]));
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n3,30\n4,40');
    expect(result.current.inputData).toBe('[{"a":1},{"a":2}]');
  });

  it.each(['success', 'failure'] as const)('ignores an obsolete %s after clearing and submitting new input', async outcome => {
    // Reset must retire the old request, including its history and finally writes.
    const first = deferred<PredictionResponse>();
    const second = deferred<PredictionResponse>();
    vi.spyOn(deploymentApi, 'predict').mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment).not.toBeNull());
    let obsolete!: Promise<void>;
    act(() => { obsolete = result.current.handlePredict(); });
    act(() => { result.current.handleClearInput(); });
    expect(result.current.activeRun).toBeNull();
    expect(result.current.parsedInputRows).toEqual([]);
    act(() => { result.current.setInputData('[{"a":9}]'); });
    let current!: Promise<void>;
    act(() => { current = result.current.handlePredict(); });
    const currentId = result.current.activeRun?.runId;
    await act(async () => {
      if (outcome === 'success') first.resolve({ predictions: [10, 20], model_version: 'old' });
      else first.reject(new Error('obsolete failure'));
      await obsolete;
    });
    expect(result.current.activeRun?.runId).toBe(currentId);
    expect(result.current.predictions).toBeNull();
    expect(result.current.error).toBeNull();
    expect(JSON.parse(localStorage.getItem(LS_PENDING_RUN)!).runId).toBe(currentId);
    await act(async () => { second.resolve({ predictions: [90], model_version: 'new' }); await current; });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n9,90');
    expect(result.current.runHistory).toHaveLength(1);
  });

  it('does not persist a late response after the page is unmounted', async () => {
    // Unmounted requests cannot replace another page instance's durable result.
    const response = deferred<PredictionResponse>();
    vi.spyOn(deploymentApi, 'predict').mockReturnValue(response.promise);
    const { result, unmount } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment).not.toBeNull());
    let pending!: Promise<void>;
    act(() => { pending = result.current.handlePredict(); });
    const marker = localStorage.getItem(LS_PENDING_RUN);
    unmount();
    await act(async () => { response.resolve({ predictions: [10, 20], model_version: 'old' }); await pending; });
    expect(localStorage.getItem(LS_RUN_HISTORY)).toBeNull();
    expect(localStorage.getItem(LS_PENDING_RUN)).toBe(marker);
  });

  it('records cancellation even if a response resolves after its signal was aborted', async () => {
    // An already queued success cannot turn a cancelled request into a result.
    const response = deferred<PredictionResponse>();
    vi.spyOn(deploymentApi, 'predict').mockReturnValue(response.promise);
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment).not.toBeNull());
    let pending!: Promise<void>;
    act(() => { pending = result.current.handlePredict(); });
    act(() => { result.current.handleCancelRun(); });
    await act(async () => { response.resolve({ predictions: [10, 20], model_version: 'old' }); await pending; });
    expect(result.current.currentRunMeta?.status).toBe('cancelled');
    expect(result.current.predictions).toBeNull();
    expect(result.current.parsedInputRows).toEqual([]);
  });

  it('clears failed results and restores an older successful snapshot', async () => {
    // Failed reruns have no export; history restoration must recover its own rows.
    vi.spyOn(deploymentApi, 'predict').mockResolvedValueOnce({ predictions: [10, 20], model_version: 'v1' })
      .mockRejectedValueOnce(new Error('Prediction unavailable'));
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.activeDeployment).not.toBeNull());
    await act(async () => { await result.current.handlePredict(); });
    const successful = result.current.currentRunMeta!;
    act(() => { result.current.setInputData('[{"a":9}]'); });
    await act(async () => { await result.current.handlePredict(); });
    expect(result.current.predictions).toBeNull();
    expect(result.current.parsedInputRows).toEqual([]);
    act(() => { result.current.handleRestoreRun(successful); });
    act(() => { result.current.setInputData('[{"a":99}]'); });
    act(() => { result.current.handleDownloadCsv(); });
    expect(await readBlob(exported)).toBe('a,prediction\n1,10\n2,20');
    expect(result.current.error).toBeNull();
  });

  it('retires a pending request when its deployment changes', async () => {
    // A former deployment cannot publish results or clear the new job's marker.
    const first = deferred<PredictionResponse>();
    const second = deferred<PredictionResponse>();
    vi.spyOn(deploymentApi, 'predict').mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const deployment = {
      id: 1, job_id: 'job-a', model_type: 'regressor', artifact_uri: 'model.joblib',
      is_active: true, created_at: '2026-09-12T00:00:00Z',
    };
    const { result, rerender } = renderHook(({ jobId }) => useInferenceRuns(
      { ...deployment, job_id: jobId }, null, () => {},
    ), { initialProps: { jobId: 'job-a' } });
    let obsolete!: Promise<void>;
    act(() => { obsolete = result.current.submitRun([{ a: 1 }], null, '[{"a":1}]'); });
    rerender({ jobId: 'job-b' });
    let current!: Promise<void>;
    act(() => { current = result.current.submitRun([{ a: 9 }], null, '[{"a":9}]'); });
    await act(async () => { first.resolve({ predictions: [10], model_version: 'old' }); await obsolete; });
    expect(result.current.predictions).toBeNull();
    expect(result.current.activeRun).not.toBeNull();
    await act(async () => { second.resolve({ predictions: [90], model_version: 'new' }); await current; });
    expect(result.current.currentRunMeta).toMatchObject({ jobId: 'job-b', input: '[{"a":9}]', predictions: [90] });
    expect(result.current.runHistory).toHaveLength(1);
  });
});
