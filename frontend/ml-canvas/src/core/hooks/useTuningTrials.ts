import { useEffect, useState } from 'react';
import { JobInfo, jobsApi } from '../api/jobs';
import { JobEvent, jobEventsSocket } from '../realtime/jobEventsSocket';
import { isTerminalStatus } from './useJobPolling';

/**
 * Live series for the trial/iteration chart in JobDetailsView.
 *
 * Two event kinds feed the chart: `trial` events (tuning search progress)
 * and `iteration` events (per-round boosting progress, emitted for the
 * final refit pass). A boosting tuning job produces BOTH series (trials
 * first, refit iterations after), so the hook keeps two independent
 * slices and reports which one is currently streaming (`activeKind`);
 * the view shows them behind tabs with auto-follow + manual pin.
 * Running jobs accumulate points from WebSocket events; a one-shot
 * snapshot (`GET /jobs/{id}/trials`) backfills what completed before the
 * page subscribed. Terminal jobs redraw from persisted `metrics.trials` /
 * `metrics.iterations`, which are always complete and win over live
 * points per slice. Best-so-far runs max for maximize metrics and min
 * for minimize metrics (boosting scores are usually losses).
 */

export interface TrialPoint {
  trial: number;
  score: number;
  best: number;
}

export type SeriesKind = 'trial' | 'iteration';
export type SeriesDirection = 'minimize' | 'maximize';

export interface SeriesSlice {
  points: TrialPoint[];
  latest?: { trial: number; total: number } | undefined;
  metric?: string | undefined;
  direction: SeriesDirection;
}

export interface TuningSeriesState {
  trial: SeriesSlice;
  iteration: SeriesSlice;
  /** The series currently streaming: live = kind of the latest received
   * event; terminal = iterations when persisted, else trials. */
  activeKind: SeriesKind;
}

// Bound the in-memory series so a pathological point count cannot grow
// the chart forever; the tail (most recent points) is what matters.
const MAX_POINTS = 2000;

function isFiniteScore(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function advanceBest(best: number, score: number, direction: SeriesDirection): number {
  return direction === 'maximize' ? Math.max(best, score) : Math.min(best, score);
}

/** Unions two point sets by x index (snapshot + live events can race),
 * sorts, recomputes the running best for the given direction, and caps. */
export function mergePoints(
  existing: TrialPoint[],
  incoming: Array<{ trial: number; score: number }>,
  direction: SeriesDirection,
): TrialPoint[] {
  const byTrial = new Map<number, number>();
  for (const p of existing) byTrial.set(p.trial, p.score);
  for (const p of incoming) byTrial.set(p.trial, p.score);
  const trials = [...byTrial.keys()].sort((a, b) => a - b);
  const points: TrialPoint[] = [];
  let best = direction === 'maximize' ? -Infinity : Infinity;
  for (const trial of trials) {
    const score = byTrial.get(trial)!;
    best = advanceBest(best, score, direction);
    points.push({ trial, score, best });
  }
  return points.slice(-MAX_POINTS);
}

/** Trial-series merge helper (trials are always higher-is-better). */
export function mergeTrialPoints(
  existing: TrialPoint[],
  incoming: Array<{ trial: number; score: number }>,
): TrialPoint[] {
  return mergePoints(existing, incoming, 'maximize');
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

interface RawIterationEntry {
  iteration?: unknown;
  score?: unknown;
  direction?: unknown;
}

function isMinimize(value: unknown): boolean {
  return value !== 'maximize';
}

/** Resolves the series direction from raw iteration entries (the adapter
 * records it per point); defaults to minimize — boosting scores are losses. */
function resolveDirection(entries: RawIterationEntry[]): SeriesDirection {
  for (let i = entries.length - 1; i >= 0; i -= 1) {
    const d = entries[i]?.direction;
    if (d === 'minimize' || d === 'maximize') return d;
  }
  return 'minimize';
}

/** Builds chart points from persisted `metrics.iterations`
 * (`[{ iteration, total, score, metric, direction }, ...]`), list order =
 * x index, running best per direction, skipping non-finite scores. */
export function buildIterationSeries(
  iterations: unknown[],
  direction?: SeriesDirection,
): TrialPoint[] {
  const entries = iterations as RawIterationEntry[];
  const dir = direction ?? resolveDirection(entries);
  const points: TrialPoint[] = [];
  let best = dir === 'maximize' ? -Infinity : Infinity;
  for (const entry of entries) {
    const score = entry?.score;
    if (!isFiniteScore(score)) continue;
    const x = typeof entry?.iteration === 'number' ? entry.iteration : points.length + 1;
    best = advanceBest(best, score, dir);
    points.push({ trial: x, score, best });
  }
  return points.slice(-MAX_POINTS);
}

function readMetricsRecord(job: JobInfo | null | undefined): Record<string, unknown> {
  // `metrics` is typed `Record<string, number>` but the tuning runner
  // persists `trials`/`iterations` arrays inside it — same cast
  // JobDetailsView uses.
  return (job?.metrics as unknown as Record<string, unknown> | undefined) ?? {};
}

function readPersistedTrials(job: JobInfo | null | undefined): unknown[] {
  const trials = readMetricsRecord(job).trials;
  return Array.isArray(trials) ? trials : [];
}

function readPersistedIterations(job: JobInfo | null | undefined): unknown[] {
  const iterations = readMetricsRecord(job).iterations;
  return Array.isArray(iterations) ? iterations : [];
}

function readPersistedIterationDirection(job: JobInfo | null | undefined): SeriesDirection {
  const record = readMetricsRecord(job);
  const top = record.iteration_direction;
  if (top === 'minimize' || top === 'maximize') return top;
  return resolveDirection(readPersistedIterations(job) as RawIterationEntry[]);
}

function readPersistedIterationMetric(job: JobInfo | null | undefined): string | undefined {
  const record = readMetricsRecord(job);
  if (typeof record.iteration_metric === 'string') return record.iteration_metric;
  const iterations = readPersistedIterations(job) as Array<{ metric?: unknown }>;
  for (let i = iterations.length - 1; i >= 0; i -= 1) {
    const m = iterations[i]?.metric;
    if (typeof m === 'string') return m;
  }
  return undefined;
}

export function useTuningTrials(job: JobInfo | null | undefined): TuningSeriesState {
  const jobId = job?.job_id ?? '';
  const terminal = isTerminalStatus(job?.status);
  const [liveTrialPoints, setLiveTrialPoints] = useState<TrialPoint[]>([]);
  const [liveIterationPoints, setLiveIterationPoints] = useState<TrialPoint[]>([]);
  const [latestTrial, setLatestTrial] = useState<{ trial: number; total: number } | undefined>(
    undefined,
  );
  const [latestIteration, setLatestIteration] = useState<
    { trial: number; total: number } | undefined
  >(undefined);
  const [trialMetric, setTrialMetric] = useState<string | undefined>(undefined);
  const [iterationMetric, setIterationMetric] = useState<string | undefined>(undefined);
  const [iterationDirection, setIterationDirection] = useState<SeriesDirection>('minimize');
  const [lastEventKind, setLastEventKind] = useState<SeriesKind>('trial');

  // Fresh series per job.
  useEffect(() => {
    setLiveTrialPoints([]);
    setLiveIterationPoints([]);
    setLatestTrial(undefined);
    setLatestIteration(undefined);
    setTrialMetric(undefined);
    setIterationMetric(undefined);
    setIterationDirection('minimize');
    setLastEventKind('trial');
  }, [jobId]);

  // Backfill the points that completed before this page subscribed; the
  // live socket below only delivers events from subscription onward.
  // Both series are kept — a boosting tuning job has trials AND refit
  // iterations.
  useEffect(() => {
    if (!jobId || terminal) return;
    let cancelled = false;
    jobsApi
      .getJobTrials(jobId)
      .then((snapshot) => {
        if (cancelled) return;
        const iterations = (snapshot.iterations ?? []).filter((t) => isFiniteScore(t.score));
        if (iterations.length > 0) {
          const direction = resolveDirection(iterations);
          setIterationDirection(direction);
          setLiveIterationPoints((prev) =>
            mergePoints(
              prev,
              iterations.map((t) => ({ trial: t.iteration, score: t.score })),
              direction,
            ),
          );
          const last = iterations[iterations.length - 1]!;
          setLatestIteration({ trial: last.iteration, total: last.total });
          setIterationMetric(snapshot.iteration_metric ?? last.metric ?? undefined);
          setLastEventKind('iteration');
        }
        const entries = snapshot.trials.filter((t) => isFiniteScore(t.score));
        if (entries.length > 0) {
          setLiveTrialPoints((prev) => mergeTrialPoints(prev, entries));
          const last = entries[entries.length - 1]!;
          setLatestTrial({ trial: last.trial, total: last.total });
          if (iterations.length === 0) setLastEventKind('trial');
        }
        if (snapshot.metric) setTrialMetric(snapshot.metric);
      })
      .catch(() => {
        // Backfill is best-effort; the live stream still works without it.
      });
    return () => {
      cancelled = true;
    };
  }, [jobId, terminal]);

  // Live accumulation only while the job can still emit points.
  useEffect(() => {
    if (!jobId || terminal) return undefined;
    const unsubscribe = jobEventsSocket.subscribe((evt: JobEvent) => {
      if (evt.job_id !== jobId) return;
      if (evt.event === 'trial') {
        if (!isFiniteScore(evt.trial_score)) return;
        const score = evt.trial_score;
        const trial = evt.trial_number;
        setLiveTrialPoints((prev) =>
          mergeTrialPoints(prev, [{ trial: trial ?? prev.length + 1, score }]),
        );
        if (trial !== undefined && evt.trial_total !== undefined) {
          setLatestTrial({ trial, total: evt.trial_total });
        }
        if (evt.trial_metric) setTrialMetric(evt.trial_metric);
        setLastEventKind('trial');
      } else if (evt.event === 'iteration') {
        if (!isFiniteScore(evt.iteration_score)) return;
        const score = evt.iteration_score;
        const iteration = evt.iteration_number;
        const direction: SeriesDirection = isMinimize(evt.iteration_direction)
          ? 'minimize'
          : 'maximize';
        setIterationDirection(direction);
        setLiveIterationPoints((prev) =>
          mergePoints(prev, [{ trial: iteration ?? prev.length + 1, score }], direction),
        );
        if (iteration !== undefined && evt.iteration_total !== undefined) {
          setLatestIteration({ trial: iteration, total: evt.iteration_total });
        }
        if (evt.iteration_metric) setIterationMetric(evt.iteration_metric);
        setLastEventKind('iteration');
      }
    });
    return unsubscribe;
  }, [jobId, terminal]);

  const persistedIterations = buildIterationSeries(
    readPersistedIterations(job),
    readPersistedIterationDirection(job),
  );
  const persistedTrials = buildTrialSeries(readPersistedTrials(job));

  // Per slice: a terminal job's persisted list is complete and
  // authoritative (live points only saw events since subscription);
  // otherwise the live accumulation is the series.
  const trialSlice: SeriesSlice = {
    points: terminal && persistedTrials.length > 0 ? persistedTrials : liveTrialPoints,
    latest: latestTrial,
    metric: trialMetric,
    direction: 'maximize',
  };
  const iterationsFromPersisted = terminal && persistedIterations.length > 0;
  const iterationSlice: SeriesSlice = {
    points: iterationsFromPersisted ? persistedIterations : liveIterationPoints,
    latest: latestIteration,
    metric: iterationMetric ?? readPersistedIterationMetric(job),
    direction: iterationsFromPersisted ? readPersistedIterationDirection(job) : iterationDirection,
  };
  const activeKind: SeriesKind = terminal
    ? persistedIterations.length > 0
      ? 'iteration'
      : 'trial'
    : lastEventKind;

  return { trial: trialSlice, iteration: iterationSlice, activeKind };
}
