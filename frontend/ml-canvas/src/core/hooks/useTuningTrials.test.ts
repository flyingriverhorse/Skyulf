import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { JobEvent } from '../realtime/jobEventsSocket';
import { jobEventsSocket } from '../realtime/jobEventsSocket';
import type { JobInfo } from '../api/jobs';
import { jobsApi } from '../api/jobs';
import { buildIterationSeries, buildTrialSeries, mergeTrialPoints, useTuningTrials } from './useTuningTrials';

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

const emitIteration = (event: Partial<JobEvent>) => {
  const subscribeMock = jobEventsSocket.subscribe as ReturnType<typeof vi.fn>;
  const handler = subscribeMock.mock.calls.at(-1)?.[0] as
    | ((evt: JobEvent) => void)
    | undefined;
  expect(handler).toBeDefined();
  act(() => {
    handler!({
      event: 'iteration',
      job_id: 'job-1',
      iteration_direction: 'minimize',
      ...event,
    } as JobEvent);
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
    expect(result.current.trial.points).toHaveLength(2);
    expect(result.current.trial.points[1]).toEqual({ trial: 2, score: 0.8, best: 0.8 });
    expect(result.current.activeKind).toBe('trial');
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
    expect(result.current.trial.points.map((p) => p.trial)).toEqual([1, 2, 3]);
    expect(result.current.trial.points.map((p) => p.best)).toEqual([0.5, 0.8, 0.8]);
    expect(result.current.trial.latest).toEqual({ trial: 3, total: 5 });
    expect(result.current.trial.metric).toBe('accuracy');

    // Live events continue from the snapshot without duplicating trial 3.
    emit({ trial_number: 3, trial_total: 5, trial_score: 0.7 });
    emit({ trial_number: 4, trial_total: 5, trial_score: 0.9 });
    expect(result.current.trial.points.map((p) => p.trial)).toEqual([1, 2, 3, 4]);
    expect(result.current.trial.points[3]).toEqual({ trial: 4, score: 0.9, best: 0.9 });
  });

  it('swallows snapshot failures and still streams live events', async () => {
    snapshotMock.mockRejectedValue(new Error('boom'));
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    expect(result.current.trial.points).toEqual([]);
    emit({ trial_number: 1, trial_total: 2, trial_score: 0.5 });
    expect(result.current.trial.points).toHaveLength(1);
  });

  it('accumulates live trial events with a monotone best', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emit({ trial_number: 1, trial_total: 3, trial_score: 0.5, trial_metric: 'accuracy' });
    emit({ trial_number: 2, trial_total: 3, trial_score: 0.9, trial_metric: 'accuracy' });
    emit({ trial_number: 3, trial_total: 3, trial_score: 0.7, trial_metric: 'accuracy' });
    expect(result.current.trial.points.map((p) => p.score)).toEqual([0.5, 0.9, 0.7]);
    expect(result.current.trial.points.map((p) => p.best)).toEqual([0.5, 0.9, 0.9]);
    expect(result.current.trial.latest).toEqual({ trial: 3, total: 3 });
    expect(result.current.trial.metric).toBe('accuracy');
  });

  it('ignores foreign jobs, non-trial events, and scoreless trials', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emit({ job_id: 'other', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ event: 'progress', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ trial_number: 1, trial_total: 2 });
    expect(result.current.trial.points).toEqual([]);
    expect(result.current.trial.latest).toBeUndefined();
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
    expect(result.current.trial.points).toHaveLength(2);

    rerender({
      j: job({
        status: 'completed',
        metrics: {
          trials: Array.from({ length: 5 }, (_, i) => ({ params: {}, score: 0.5 + i * 0.05 })),
        } as unknown as Record<string, number>,
      }),
    });
    // Full curve from trial 1, not the 2-point live tail.
    expect(result.current.trial.points.map((p) => p.trial)).toEqual([1, 2, 3, 4, 5]);
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
    expect(result.current.trial.points).toHaveLength(1);
    rerender({ j: job({ job_id: 'job-2' }) });
    await flush();
    expect(result.current.trial.points).toEqual([]);
    expect(result.current.trial.latest).toBeUndefined();
  });
});

describe('buildIterationSeries', () => {
  it('builds points with a running-min best under minimize', () => {
    const iterations = [
      { iteration: 1, total: 3, score: 0.6, metric: 'logloss', direction: 'minimize' },
      { iteration: 2, total: 3, score: 0.4, metric: 'logloss', direction: 'minimize' },
      { iteration: 3, total: 3, score: 0.45, metric: 'logloss', direction: 'minimize' },
    ];
    expect(buildIterationSeries(iterations)).toEqual([
      { trial: 1, score: 0.6, best: 0.6 },
      { trial: 2, score: 0.4, best: 0.4 },
      { trial: 3, score: 0.45, best: 0.4 },
    ]);
  });

  it('honors an explicit maximize direction', () => {
    const iterations = [
      { iteration: 1, total: 2, score: 0.5, metric: 'auc', direction: 'maximize' },
      { iteration: 2, total: 2, score: 0.4, metric: 'auc', direction: 'maximize' },
    ];
    expect(buildIterationSeries(iterations).map((p) => p.best)).toEqual([0.5, 0.5]);
  });

  it('skips non-finite scores', () => {
    const iterations = [
      { iteration: 1, total: 3, score: 0.5, direction: 'minimize' },
      { iteration: 2, total: 3, score: Number.NaN, direction: 'minimize' },
      { iteration: 3, total: 3, score: 0.7, direction: 'minimize' },
    ];
    expect(buildIterationSeries(iterations)).toEqual([
      { trial: 1, score: 0.5, best: 0.5 },
      { trial: 3, score: 0.7, best: 0.5 },
    ]);
  });
});

describe('useTuningTrials — boosting iterations', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (jobEventsSocket.subscribe as ReturnType<typeof vi.fn>).mockReturnValue(() => {});
    snapshotMock.mockResolvedValue({ trials: [], metric: null });
  });

  it('accumulates iteration events with a decreasing best under minimize', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emitIteration({ iteration_number: 1, iteration_total: 200, iteration_score: 0.6, iteration_metric: 'logloss' });
    emitIteration({ iteration_number: 2, iteration_total: 200, iteration_score: 0.4 });
    emitIteration({ iteration_number: 3, iteration_total: 200, iteration_score: 0.45 });
    expect(result.current.activeKind).toBe('iteration');
    expect(result.current.iteration.direction).toBe('minimize');
    expect(result.current.iteration.points.map((p) => p.score)).toEqual([0.6, 0.4, 0.45]);
    expect(result.current.iteration.points.map((p) => p.best)).toEqual([0.6, 0.4, 0.4]);
    expect(result.current.iteration.latest).toEqual({ trial: 3, total: 200 });
    expect(result.current.iteration.metric).toBe('logloss');
  });

  it('populates both slices from mixed events and follows the latest kind', async () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    emit({ trial_number: 1, trial_total: 3, trial_score: 0.7, trial_metric: 'accuracy' });
    emit({ trial_number: 2, trial_total: 3, trial_score: 0.8 });
    expect(result.current.activeKind).toBe('trial');
    expect(result.current.trial.points).toHaveLength(2);
    expect(result.current.iteration.points).toHaveLength(0);

    // The refit starts streaming: trials keep their slice, iterations grow
    // theirs, and the active series passes to iterations.
    emitIteration({ iteration_number: 1, iteration_total: 100, iteration_score: 0.5 });
    emitIteration({ iteration_number: 2, iteration_total: 100, iteration_score: 0.4 });
    expect(result.current.activeKind).toBe('iteration');
    expect(result.current.trial.points).toHaveLength(2);
    expect(result.current.iteration.points.map((p) => p.trial)).toEqual([1, 2]);
    expect(result.current.iteration.latest).toEqual({ trial: 2, total: 100 });
    expect(result.current.trial.latest).toEqual({ trial: 2, total: 3 });
  });

  it('backfills iterations from the snapshot when opened mid-refit', async () => {
    snapshotMock.mockResolvedValue({
      trials: [{ trial: 1, total: 1, score: 0.7, metric: 'accuracy' }],
      metric: 'accuracy',
      iterations: [
        { iteration: 1, total: 10, score: 0.5, metric: 'logloss', direction: 'minimize' },
        { iteration: 2, total: 10, score: 0.3, metric: 'logloss', direction: 'minimize' },
      ],
      iteration_metric: 'logloss',
    });
    const { result } = renderHook(() => useTuningTrials(job()));
    await flush();
    expect(result.current.activeKind).toBe('iteration');
    expect(result.current.iteration.points.map((p) => p.best)).toEqual([0.5, 0.3]);
    expect(result.current.iteration.latest).toEqual({ trial: 2, total: 10 });
    expect(result.current.iteration.metric).toBe('logloss');
    expect(result.current.trial.points).toHaveLength(1);
    expect(result.current.trial.metric).toBe('accuracy');

    // Live iterations continue without duplicating iteration 2.
    emitIteration({ iteration_number: 2, iteration_total: 10, iteration_score: 0.3 });
    emitIteration({ iteration_number: 3, iteration_total: 10, iteration_score: 0.25 });
    expect(result.current.iteration.points.map((p) => p.trial)).toEqual([1, 2, 3]);
    expect(result.current.iteration.points[2]).toEqual({ trial: 3, score: 0.25, best: 0.25 });
  });

  it('keeps both persisted series once terminal and reports iterations active', async () => {
    const completed = job({
      status: 'completed',
      metrics: {
        trials: [{ params: {}, score: 0.8 }, { params: {}, score: 0.85 }],
        iterations: [
          { iteration: 1, total: 3, score: 0.6, direction: 'minimize' },
          { iteration: 2, total: 3, score: 0.4, direction: 'minimize' },
          { iteration: 3, total: 3, score: 0.5, direction: 'minimize' },
        ],
        iteration_direction: 'minimize',
        iteration_metric: 'logloss',
      } as unknown as Record<string, number>,
    });
    const { result } = renderHook(() => useTuningTrials(completed));
    await flush();
    expect(result.current.activeKind).toBe('iteration');
    expect(result.current.iteration.points).toHaveLength(3);
    expect(result.current.iteration.points.map((p) => p.best)).toEqual([0.6, 0.4, 0.4]);
    expect(result.current.iteration.metric).toBe('logloss');
    expect(result.current.trial.points).toHaveLength(2);
    expect(result.current.trial.points[1]).toEqual({ trial: 2, score: 0.85, best: 0.85 });
  });
});
