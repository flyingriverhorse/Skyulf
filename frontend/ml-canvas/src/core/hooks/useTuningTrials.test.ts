import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type { JobEvent } from '../realtime/jobEventsSocket';
import { jobEventsSocket } from '../realtime/jobEventsSocket';
import type { JobInfo } from '../api/jobs';
import { buildTrialSeries, useTuningTrials } from './useTuningTrials';

vi.mock('../realtime/jobEventsSocket', () => ({
  jobEventsSocket: {
    subscribe: vi.fn(() => () => {}),
    onStatus: vi.fn(() => () => {}),
  },
}));

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

describe('useTuningTrials', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (jobEventsSocket.subscribe as ReturnType<typeof vi.fn>).mockReturnValue(() => {});
  });

  it('seeds points from persisted metrics.trials for completed jobs', () => {
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
    expect(result.current.points).toHaveLength(2);
    expect(result.current.points[1]).toEqual({ trial: 2, score: 0.8, best: 0.8 });
  });

  it('accumulates live trial events with a monotone best', () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    emit({ trial_number: 1, trial_total: 3, trial_score: 0.5, trial_metric: 'accuracy' });
    emit({ trial_number: 2, trial_total: 3, trial_score: 0.9, trial_metric: 'accuracy' });
    emit({ trial_number: 3, trial_total: 3, trial_score: 0.7, trial_metric: 'accuracy' });
    expect(result.current.points.map((p) => p.score)).toEqual([0.5, 0.9, 0.7]);
    expect(result.current.points.map((p) => p.best)).toEqual([0.5, 0.9, 0.9]);
    expect(result.current.latest).toEqual({ trial: 3, total: 3 });
    expect(result.current.metric).toBe('accuracy');
  });

  it('ignores foreign jobs, non-trial events, and scoreless trials', () => {
    const { result } = renderHook(() => useTuningTrials(job()));
    emit({ job_id: 'other', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ event: 'progress', trial_number: 1, trial_total: 2, trial_score: 0.9 });
    emit({ trial_number: 1, trial_total: 2 });
    expect(result.current.points).toEqual([]);
    expect(result.current.latest).toBeUndefined();
  });

  it('unsubscribes on unmount', () => {
    const unsubscribe = vi.fn();
    (jobEventsSocket.subscribe as ReturnType<typeof vi.fn>).mockReturnValue(unsubscribe);
    const { unmount } = renderHook(() => useTuningTrials(job()));
    unmount();
    expect(unsubscribe).toHaveBeenCalled();
  });

  it('does not subscribe for terminal jobs', () => {
    renderHook(() => useTuningTrials(job({ status: 'completed' })));
    expect(jobEventsSocket.subscribe).not.toHaveBeenCalled();
  });

  it('resets when the job changes', () => {
    const { result, rerender } = renderHook(
      ({ j }: { j: JobInfo }) => useTuningTrials(j),
      { initialProps: { j: job() } },
    );
    emit({ trial_number: 1, trial_total: 2, trial_score: 0.5 });
    expect(result.current.points).toHaveLength(1);
    rerender({ j: job({ job_id: 'job-2' }) });
    expect(result.current.points).toEqual([]);
    expect(result.current.latest).toBeUndefined();
  });
});
