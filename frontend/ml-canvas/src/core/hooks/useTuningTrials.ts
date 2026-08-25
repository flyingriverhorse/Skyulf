import { useEffect, useState } from 'react';
import { JobInfo } from '../api/jobs';
import { JobEvent, jobEventsSocket } from '../realtime/jobEventsSocket';
import { isTerminalStatus } from './useJobPolling';

/**
 * Live tuning-trial series for the trial chart in JobDetailsView.
 *
 * Running tuning jobs accumulate points from `trial` WebSocket events;
 * terminal jobs redraw the same series from the persisted
 * `metrics.trials` the backend already saves. Best-so-far assumes
 * higher-is-better, matching the tuning engine's own tracking.
 */

export interface TrialPoint {
  trial: number;
  score: number;
  best: number;
}

export interface TrialSeriesState {
  points: TrialPoint[];
  latest?: { trial: number; total: number } | undefined;
  metric?: string | undefined;
}

// Bound the in-memory series so a pathological trial count cannot grow
// the chart forever; the tail (most recent trials) is what matters.
const MAX_POINTS = 2000;

function isFiniteScore(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/** Builds chart points (1-based completion index + running-max best) from
 * persisted `[{ params, score }, ...]` trials, skipping non-finite scores. */
export function buildTrialSeries(trials: unknown[]): TrialPoint[] {
  const points: TrialPoint[] = [];
  let best = -Infinity;
  for (const trial of trials) {
    const score = (trial as { score?: unknown } | null | undefined)?.score;
    if (!isFiniteScore(score)) continue;
    best = Math.max(best, score);
    points.push({ trial: points.length + 1, score, best });
  }
  return points.slice(-MAX_POINTS);
}

function readPersistedTrials(job: JobInfo | null | undefined): unknown[] {
  if (!job?.metrics) return [];
  // `metrics` is typed `Record<string, number>` but the tuning runner
  // persists a `trials` array inside it — same cast JobDetailsView uses.
  const trials = (job.metrics as unknown as Record<string, unknown>).trials;
  return Array.isArray(trials) ? trials : [];
}

export function useTuningTrials(job: JobInfo | null | undefined): TrialSeriesState {
  const jobId = job?.job_id ?? '';
  const terminal = isTerminalStatus(job?.status);
  const [livePoints, setLivePoints] = useState<TrialPoint[]>([]);
  const [latest, setLatest] = useState<{ trial: number; total: number } | undefined>(undefined);
  const [metric, setMetric] = useState<string | undefined>(undefined);

  // Fresh series per job.
  useEffect(() => {
    setLivePoints([]);
    setLatest(undefined);
    setMetric(undefined);
  }, [jobId]);

  // Live accumulation only while the job can still emit trials.
  useEffect(() => {
    if (!jobId || terminal) return undefined;
    const unsubscribe = jobEventsSocket.subscribe((evt: JobEvent) => {
      if (evt.event !== 'trial' || evt.job_id !== jobId) return;
      if (!isFiniteScore(evt.trial_score)) return;
      const score = evt.trial_score;
      setLivePoints((prev) => {
        const best = prev.length > 0 ? Math.max(prev[prev.length - 1]!.best, score) : score;
        return [...prev, { trial: evt.trial_number ?? prev.length + 1, score, best }].slice(
          -MAX_POINTS,
        );
      });
      if (evt.trial_number !== undefined && evt.trial_total !== undefined) {
        setLatest({ trial: evt.trial_number, total: evt.trial_total });
      }
      if (evt.trial_metric) setMetric(evt.trial_metric);
    });
    return unsubscribe;
  }, [jobId, terminal]);

  const points = livePoints.length > 0 ? livePoints : buildTrialSeries(readPersistedTrials(job));
  return { points, latest, metric };
}
