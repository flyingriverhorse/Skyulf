import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { JobEvent } from '../realtime/jobEventsSocket';
import { jobEventsSocket } from '../realtime/jobEventsSocket';
import type { JobInfo } from '../api/jobs';
import { jobsApi } from '../api/jobs';
import { buildTrialSeries, mergeTrialPoints, useTuningTrials } from './useTuningTrials';

vi.mock('../realtime/jobEventsSocket', () => ({
  jobEventsSocket: {
    subscribe: vi.fn(() => () => {}),
    onStatus: vi.fn(() => () => {}),
  },
}));

vi.mock('../api/jobs', async () => {
  const actual = await vi.importActual<typeof import('../api/jobs')>('../api/jobs');
  return {
    ...actual,
    jobsApi: {
      ...actual.jobsApi,
      getJobTrials: vi.fn().mockResolvedValue({ trials: [], metric: null }),
    },
  };
});

const snapshotMock = jobsApi.getJobTrials as ReturnType<typeof vi.fn>;

const job = (overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: 'job-1',
  pipeline_id: 'p',
  node_id: 'n',
  job_type: 'tuning',
  status: 'running',
  start_time: null,
  end_time: null,
  error: null,
  result: null,
  created_at: '',
  ...overrides,
});

const emit = (event: Partial<JobEvent>) => {
  const subscribeMock = jobEventsSocket.subscribe as ReturnType<typeof vi.fn>;
  const handler = subscribeMock.mock.calls.at(-1)?.[0] as
    | ((evt: JobEvent) => void)
    | undefined;
  expect(handler).toBeDefined();
  act(() => {
    handler!({ event: 'trial', job_id: 'job-1', ...event } as JobEvent);
  });
};

/** Flush the snapshot promise + its state updates. */
const flush = async () => {
  await act(async () => {});
};

describe('buildTrialSeries', () => {
  it('builds points with a running-max best from persisted trials', () => {
    const trials = [
      { params: { C: 0.1 }, score: 0.6 },
      { params: { C: 1 }, score: 0.8 },
      { params: { C: 10 }, score: 0.7 },
    ];
    expect(buildTrialSeries(trials)).toEqual([
      { trial: 1, score: 0.6, best: 0.6 },
      { trial: 2, score: 0.8, best: 0.8 },
      { trial: 3, score: 0.7, best: 0.8 },
    ]);
  });

  it('filters null, NaN, and non-numeric scores', () => {
    const trials = [
      { params: {}, score: 0.5 },
      { params: {}, score: null },
      { params: {}, score: Number.NaN },
      { params: {}, score: 'oops' },
      { params: {}, score: 0.9 },
    ];
    expect(buildTrialSeries(trials)).toEqual([
      { trial: 1, score: 0.5, best: 0.5 },
      { trial: 2, score: 0.9, best: 0.9 },
    ]);
  });

  it('caps at 2000 points keeping the tail with global best intact', () => {
    const trials = Array.from({ length: 2500 }, (_, i) => ({
      params: {},
      score: i === 0 ? 0.99 : 0.1,
    }));
    const points = buildTrialSeries(trials);
    expect(points).toHaveLength(2000);
    expect(points[0]!.trial).toBe(501);
    // Best was set by trial 1, which fell outside the window.
    expect(points.every((p) => p.best === 0.99)).toBe(true);
  });
});

describe('mergeTrialPoints', () => {
  it('unions by trial number, sorts, and recomputes best', () => {
    const existing = [
      { trial: 3, score: 0.7, best: 0.9 },
      { trial: 4, score: 0.6, best: 0.9 },
    ];
    const incoming = [
      { trial: 1, score: 0.5 },
      { trial: 3, score: 0.7 },
      { trial: 2, score: 0.9 },
    ];
    expect(mergeTrialPoints(existing, incoming)).toEqual([
      { trial: 1, score: 0.5, best: 0.5 },
      { trial: 2, score: 0.9, best: 0.9 },
      { trial: 3, score: 0.7, best: 0.9 },
      { trial: 4, score: 0.6, best: 0.9 },
    ]);
  });
});

describe('useTuningTrials', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (jobEventsSocket.subscribe as ReturnType<typeof vi.fn>).mockReturnValue(() => {});
    snapshotMock.mockResolvedValue({ trials: [], metric: null });
  });

  it('seeds points from persisted metrics.trials for completed jobs', async () => {
    const completed = job({
      status: 'completed',
      metrics: {
        trials: [
          { params: {}, score: 0.6 },
          { params: {}, score: 0.8 },
        ],
      } as unknown as Record<string, number>,
    });
    const { result } = renderHook(() => useTuningTrials(completed));
    await flush();
    expect(result.current.points).toHaveLength(2);
    expect(result.current.points[1]).toEqual({ trial: 2, score: 0.8, best: 0.8 });
  });

  it('backfills missed trials from the snapshot when opened mid-run', async () => {
    snapshotMock.mockResolvedValue({
      trials: [
        { trial: 1, total: 5, score: 0.5, metric: 'accuracy' },
        { trial: 2, total: 5, score: 0.8, metric: 'accuracy' },
        { trial: 3, total: 5, score: 0.7, metric: 'accuracy' },
      ],
      metric: 'accuracy',
    });
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    // Curve starts at trial 1 even though we subscribed at trial 3.
    expect(result.current.points.map((p) => p.trial)).toEqual([1, 2, 3]);
    expect(result.current.points.map((p) => p.best)).toEqual([0.5, 0.8, 0.8]);
    expect(result.current.latest).toEqual({ trial: 3, total: 5 });
    expect(result.current.metric).toBe('accuracy');

    // Live events continue from the snapshot without duplicating trial 3.
    emit({ trial_number: 3, trial_total: 5, trial_score: 0.7 });
    emit({ trial_number: 4, trial_total: 5, trial_score: 0.9 });
    expect(result.current.points.map((p) => p.trial)).toEqual([1, 2, 3, 4]);
    expect(result.current.points[3]).toEqual({ trial: 4, score: 0.9, best: 0.9 });
  });

  it('swallows snapshot failures and still streams live events', async () => {
    snapshotMock.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    expect(result.current.points).toEqual([]);
    emit({ trial_number: 1, trial_total: 2, trial_score: 0.5 });
    expect(result.current.points).toHaveLength(1);
  });

  it('accumulates live trial events with a monotone best', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emit({ trial_number: 1, trial_total: 3, trial_score: 0.5, trial_metric: 'accuracy' });
    emit({ trial_number: 2, trial_total: 3, trial_score: 0.9, trial_metric: 'accuracy' });
    emit({ trial_number: 3, trial_total: 3, trial_score: 0.7, trial_metric: 'accuracy' });
    expect(result.current.points.map((p) => p.score)).toEqual([0.5, 0.9, 0.7]);
    expect(result.current.points.map((p) => p.best)).toEqual([0.5, 0.9, 0.9]);
    expect(result.current.latest).toEqual({ trial: 3, total: 3 });
    expect(result.current.metric).toBe('accuracy');
  });

  it('ignores foreign jobs, non-trial events, and scoreless trials', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emit({ job_id: 'other', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ event: 'progress', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ trial_number: 1, trial_total: 2 });
    expect(result.current.points).toEqual([]);
    expect(result.current.latest).toBeUndefined();
  });

  it('prefers the complete persisted series once the job turns terminal', async () => {
    const { result, rerender } = renderHook(
      ({ j }: { j: JobInfo }) => useTuningTrials(j),
      { initialProps: { j: job() } },
    );
    await flush();
    // Watched only the tail live (opened at trial 4 of 5).
    emit({ trial_number: 4, trial_total: 5, trial_score: 0.6 });
    emit({ trial_number: 5, trial_total: 5, trial_score: 0.7 });
    expect(result.current.points).toHaveLength(2);

    rerender({
      j: job({
        status: 'completed',
        metrics: {
          trials: Array.from({ length: 5 }, (_, i) => ({ params: {}, score: 0.5 + i * 0.05 })),
        } as unknown as Record<string, number>,
      }),
    });
    // Full curve from trial 1, not the 2-point live tail.
    expect(result.current.points.map((p) => p.trial)).toEqual([1, 2, 3, 4, 5]);
  });

  it('unsubscribes on unmount', async () => {
    const unsubscribe = vi.fn();
    (jobEventsSocket.subscribe as ReturnType<typeof vi.fn>).mockReturnValue(unsubscribe);
    const { unmount } = renderHook(() => useTuningTrials(job()));
    unmount();
    expect(unsubscribe).toHaveBeenCalled();
  });

  it('does not subscribe for terminal jobs', async () => {
    renderHook(() => useTuningTrials(job({ status: 'completed' })));
    await flush();
    expect(jobEventsSocket.subscribe).not.toHaveBeenCalled();
    expect(snapshotMock).not.toHaveBeenCalled();
  });

  it('resets when the job changes', async () => {
    const { result, rerender } = renderHook(
      ({ j }: { j: JobInfo }) => useTuningTrials(j),
      { initialProps: { j: job() } },
    );
    await flush();
    emit({ trial_number: 1, trial_total: 2, trial_score: 0.5 });
    expect(result.current.points).toHaveLength(1);
    rerender({ j: job({ job_id: 'job-2' }) });
    await flush();
    expect(result.current.points).toEqual([]);
    expect(result.current.latest).toBeUndefined();
  });
});
