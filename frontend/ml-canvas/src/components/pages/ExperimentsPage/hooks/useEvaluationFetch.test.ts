import type { AxiosResponse } from 'axios';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { apiClient } from '../../../../core/api/client';
import { thresholdTuningApi, type SavedThresholdInfo } from '../../../../core/api/thresholdTuning';
import type { JobInfo } from '../../../../core/api/jobs';
import type { EvaluationData } from '../types';
import { useEvaluationFetch } from './useEvaluationFetch';

const okResponse = (data: unknown) => ({ data } as AxiosResponse);

const makeJob = (id: string, scoringMetric?: string): JobInfo => ({
  job_id: id,
  pipeline_id: 'p1',
  node_id: 'n1',
  job_type: 'training',
  status: 'completed',
  start_time: null,
  end_time: null,
  error: null,
  result: scoringMetric ? { scoring_metric: scoringMetric } : null,
  created_at: '2026-01-01T00:00:00Z',
});

const noSavedThresholds: SavedThresholdInfo = {
  thresholds: null,
  classes: null,
  metric: null,
  split_used: null,
  computed_at: null,
  source: null,
  enabled: false,
};

const evalFor = (tag: string) => ({ tag }) as unknown as EvaluationData;

const savedThresholds: SavedThresholdInfo = {
  thresholds: { '0': 0.4, '1': 0.6 }, classes: [0, 1], metric: 'f1',
  split_used: 'validation', computed_at: '2026-09-09T10:00:00Z', source: null, enabled: false,
};

describe('useEvaluationFetch', () => {
  it('distinguishes saved but disabled thresholds from an unsaved preview', async () => {
    /** Turning a saved set off must preserve the ability to enable it again. */
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(savedThresholds);
    const { result } = renderHook(() => useEvaluationFetch([makeJob('a')]));
    await act(async () => { await result.current.fetchEvaluationData('a'); });
    expect(result.current.useTunedThresholds).toBe(false);
    expect(result.current.hasSavedThresholds).toBe(true);
  });

  it.each(['empty', 'failed'])('resets saved state when the next job has %s thresholds', async (response) => {
    /** A prior job's saved set must not unlock an invalid toggle for another job. */
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    const getSaved = vi.spyOn(thresholdTuningApi, 'get').mockResolvedValueOnce(savedThresholds);
    if (response === 'empty') getSaved.mockResolvedValueOnce(noSavedThresholds);
    else getSaved.mockRejectedValueOnce(new Error('Threshold load failed'));
    const { result } = renderHook(() => useEvaluationFetch([makeJob('a'), makeJob('b')]));
    await act(async () => { await result.current.fetchEvaluationData('a'); });
    expect(result.current.hasSavedThresholds).toBe(true);
    await act(async () => { await result.current.fetchEvaluationData('b'); });
    expect(result.current.hasSavedThresholds).toBe(false);
    expect(result.current.tuningPreview).toBeNull();
    expect(result.current.useTunedThresholds).toBe(false);
  });

  it('ignores saved thresholds that arrive after another job was selected', async () => {
    /** A late saved-set response must not re-enable the newer job's toggle. */
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    let resolveSaved!: (value: SavedThresholdInfo) => void;
    const getSaved = vi.spyOn(thresholdTuningApi, 'get')
      .mockImplementationOnce(() => new Promise(resolve => { resolveSaved = resolve; }))
      .mockResolvedValueOnce(noSavedThresholds);
    const { result } = renderHook(() => useEvaluationFetch([makeJob('a'), makeJob('b')]));
    let loadA!: Promise<void>;
    act(() => { loadA = result.current.fetchEvaluationData('a'); });
    await waitFor(() => expect(getSaved).toHaveBeenCalledWith('a'));
    await act(async () => { await result.current.fetchEvaluationData('b'); });
    await act(async () => { resolveSaved(savedThresholds); await loadA; });
    expect(result.current.evalJobId).toBe('b');
    expect(result.current.hasSavedThresholds).toBe(false);
    expect(result.current.tuningPreview).toBeNull();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('loads evaluation data and clears the loading flag on success', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(noSavedThresholds);

    const { result } = renderHook(() => useEvaluationFetch([makeJob('a')]));

    act(() => {
      void result.current.fetchEvaluationData('a');
    });

    await waitFor(() => {
      expect(result.current.evaluationData).toEqual(evalFor('A'));
    });
    expect(result.current.isEvalLoading).toBe(false);
    expect(result.current.evalJobId).toBe('a');
    expect(result.current.evalError).toBeNull();
  });

  it('sets the error message and clears the data on failure', async () => {
    vi.spyOn(apiClient, 'get').mockRejectedValue({ response: { data: { detail: 'boom' } } });
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(noSavedThresholds);

    const { result } = renderHook(() => useEvaluationFetch([makeJob('a')]));

    act(() => {
      void result.current.fetchEvaluationData('a');
    });

    await waitFor(() => {
      expect(result.current.evalError).toBe('boom');
    });
    expect(result.current.evaluationData).toBeNull();
    expect(result.current.isEvalLoading).toBe(false);
  });

  it('defaults the metric dropdown to the job scoring metric (fallback f1_weighted for unmappable)', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(noSavedThresholds);

    const { result } = renderHook(() => useEvaluationFetch([makeJob('a', 'roc_auc'), makeJob('b', 'precision_weighted')]));

    act(() => {
      void result.current.fetchEvaluationData('a');
    });
    await waitFor(() => {
      expect(result.current.selectedThresholdMetric).toBe('f1_weighted');
    });

    act(() => {
      void result.current.fetchEvaluationData('b');
    });
    await waitFor(() => {
      expect(result.current.selectedThresholdMetric).toBe('precision');
    });
  });

  it('hydrates the tuning panel from saved server-side thresholds', async () => {
    const saved: SavedThresholdInfo = {
      thresholds: { '0': 0.4, '1': 0.6 },
      classes: [0, 1],
      metric: 'f1_weighted',
      split_used: 'test',
      computed_at: '2026-01-02T00:00:00Z',
      source: 'training',
      enabled: true,
    };
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValueOnce(saved).mockResolvedValue({
      ...saved,
      metric: 'recall',
    });

    const { result } = renderHook(() => useEvaluationFetch([makeJob('a'), makeJob('b')]));

    act(() => {
      void result.current.fetchEvaluationData('a');
    });

    await waitFor(() => {
      expect(result.current.tuningPreview).toEqual({
        thresholds: saved.thresholds,
        classes: saved.classes,
        metric: saved.metric,
        split_used: saved.split_used,
        source: 'training',
      });
    });
    // f1_weighted is not a preview-endpoint metric — the dropdown keeps its
    // default instead of blanking out / failing the next Preview.
    expect(result.current.selectedTuningMetric).toBe('f1');
    expect(result.current.useTunedThresholds).toBe(true);

    // A supported saved metric does take over the dropdown.
    act(() => {
      void result.current.fetchEvaluationData('b');
    });
    await waitFor(() => {
      expect(result.current.selectedTuningMetric).toBe('recall');
    });
    expect(result.current.tuningPreview?.source).toBe('training');
  });

  it('discards a late response for an older job instead of clobbering the newer one', async () => {
    let resolveA: () => void = () => {};
    let resolveB: () => void = () => {};
    vi.spyOn(apiClient, 'get').mockImplementation((url: string) => {
      if (url.includes('/ja/evaluation')) {
        return new Promise<AxiosResponse>((res) => {
          resolveA = () => res(okResponse(evalFor('A')));
        });
      }
      return new Promise<AxiosResponse>((res) => {
        resolveB = () => res(okResponse(evalFor('B')));
      });
    });
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(noSavedThresholds);

    const { result } = renderHook(() => useEvaluationFetch([makeJob('ja'), makeJob('jb')]));

    act(() => {
      void result.current.fetchEvaluationData('ja');
    });
    act(() => {
      void result.current.fetchEvaluationData('jb');
    });

    // B (fast) resolves first — its data is rendered.
    act(() => {
      resolveB();
    });
    await waitFor(() => {
      expect(result.current.evaluationData).toEqual(evalFor('B'));
    });
    expect(result.current.evalJobId).toBe('jb');

    // A's late response must be discarded, not rendered under B's header.
    act(() => {
      resolveA();
    });
    await waitFor(() => {
      expect(result.current.evaluationData).toEqual(evalFor('B'));
    });
    expect(result.current.evalJobId).toBe('jb');
    expect(result.current.isEvalLoading).toBe(false);
  });
});
