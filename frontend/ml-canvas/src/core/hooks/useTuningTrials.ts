import { useEffect, useState } from 'react';
import { JobInfo, jobsApi } from '../api/jobs';
import { JobEvent, jobEventsSocket } from '../realtime/jobEventsSocket';
import { isTerminalStatus } from './useJobPolling';

/**
 * Live tuning-trial series for the trial chart in JobDetailsView.
 *
 * Running tuning jobs accumulate points from `trial` WebSocket events; a
 * one-shot snapshot (`GET /jobs/{id}/trials`) backfills the trials that
 * completed before the page subscribed, so a late opener still sees the
 * curve from trial 1. Terminal jobs redraw the same series from the
 * persisted `metrics.trials` the backend already saves — and once terminal,
 * the persisted list wins over live points (it is always complete).
 * Best-so-far assumes higher-is-better, matching the tuning engine's own
 * tracking.
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

/** Unions two point sets by trial number (snapshot + live events can race),
 * sorts by trial, recomputes the running-max best, and caps the length. */
export function mergeTrialPoints(
  existing: TrialPoint[],
  incoming: Array<{ trial: number; score: number }>,
): TrialPoint[] {
  const byTrial = new Map<number, number>();
  for (const p of existing) byTrial.set(p.trial, p.score);
  for (const p of incoming) byTrial.set(p.trial, p.score);
  const trials = [...byTrial.keys()].sort((a, b) => a - b);
  const points: TrialPoint[] = [];
  let best = -Infinity;
  for (const trial of trials) {
    const score = byTrial.get(trial)!;
    best = Math.max(best, score);
    points.push({ trial, score, best });
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

  // Backfill the trials that completed before this page subscribed; the
  // live socket below only delivers events from subscription onward.
  useEffect(() => {
    if (!jobId || terminal) return;
    let cancelled = false;
    jobsApi
      .getJobTrials(jobId)
      .then((snapshot) => {
        if (cancelled) return;
        const entries = snapshot.trials.filter((t) => isFiniteScore(t.score));
        if (entries.length > 0) {
          setLivePoints((prev) => mergeTrialPoints(prev, entries));
          const last = entries[entries.length - 1]!;
          setLatest({ trial: last.trial, total: last.total });
        }
        if (snapshot.metric) setMetric(snapshot.metric);
      })
      .catch(() => {
        // Backfill is best-effort; the live stream still works without it.
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, terminal]);

  // Live accumulation only while the job can still emit trials.
  useEffect(() => {
    if (!jobId || terminal) return undefined;
    const unsubscribe = jobEventsSocket.subscribe((evt: JobEvent) => {
      if (evt.event !== 'trial' || evt.job_id !== jobId) return;
      if (!isFiniteScore(evt.trial_score)) return;
      const score = evt.trial_score;
      const trial = evt.trial_number;
      setLivePoints((prev) =>
        mergeTrialPoints(prev, [{ trial: trial ?? prev.length + 1, score }]),
      );
      if (trial !== undefined && evt.trial_total !== undefined) {
        setLatest({ trial, total: evt.trial_total });
      }
      if (evt.trial_metric) setMetric(evt.trial_metric);
    });
    return unsubscribe;
  }, [jobId, terminal]);

  const persisted = buildTrialSeries(readPersistedTrials(job));
  // Terminal jobs: the persisted list is complete and authoritative — live
  // points only ever saw the trials since subscription. Running jobs (or
  // terminal jobs without persisted trials, e.g. failed mid-run): live view.
  const points = terminal && persisted.length > 0 ? persisted : livePoints;
  return { points, latest, metric };
}
