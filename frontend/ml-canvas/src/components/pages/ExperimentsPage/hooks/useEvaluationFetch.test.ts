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
  enabled: false,
};

const evalFor = (tag: string) => ({ tag }) as unknown as EvaluationData;

describe('useEvaluationFetch', () => {
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
      enabled: true,
    };
    vi.spyOn(apiClient, 'get').mockResolvedValue(okResponse(evalFor('A')));
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue(saved);

    const { result } = renderHook(() => useEvaluationFetch([makeJob('a')]));

    act(() => {
      void result.current.fetchEvaluationData('a');
    });

    await waitFor(() => {
      expect(result.current.tuningPreview).toEqual({
        thresholds: saved.thresholds,
        classes: saved.classes,
        metric: saved.metric,
        split_used: saved.split_used,
      });
    });
    expect(result.current.selectedTuningMetric).toBe('f1_weighted');
    expect(result.current.useTunedThresholds).toBe(true);
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
