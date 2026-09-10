import { useState, useEffect } from 'react';
import { JobInfo } from '../../../../core/api/jobs';
import { useTuningTrials, type SeriesKind } from '../../../../core/hooks/useTuningTrials';
import { isTerminalStatus } from '../../../../core/hooks/useJobPolling';
import { TuningTrialsChart } from '../TuningTrialsChart';
import { getScoringMetric } from './scoring';

export function useJobChart(job: JobInfo) {
  // Live tuning series: trials (search progress) and iterations (boosting
  // refit progress) stream as two independent slices; a tab row shows both
  // when a job has each, auto-following the streaming series until the
  // user pins one by clicking.
  const { trial: trialSlice, iteration: iterationSlice, activeKind } = useTuningTrials(job);
  const [pinnedSeries, setPinnedSeries] = useState<SeriesKind | null>(null);
  useEffect(() => {
    setPinnedSeries(null);
  }, [job.job_id]);
  const hasBothSeries = trialSlice.points.length >= 2 && iterationSlice.points.length >= 2;
  let visibleKind: SeriesKind = pinnedSeries ?? activeKind;
  // The chart needs >=2 points; if the picked slice is still empty-ish,
  // show the other one so the first streaming series is visible at once.
  const slices = { trial: trialSlice, iteration: iterationSlice };
  const otherKind = visibleKind === 'iteration' ? 'trial' : 'iteration';
  if (slices[visibleKind].points.length < 2 && slices[otherKind].points.length >= 2) {
    visibleKind = otherKind;
  }
  const visibleSlice = slices[visibleKind];
  return { hasBothSeries, visibleKind, visibleSlice, setPinnedSeries };
}

export function JobChart({ job, chart }: { job: JobInfo; chart: ReturnType<typeof useJobChart> }) {
  const { hasBothSeries, visibleKind, visibleSlice, setPinnedSeries } = chart;
  return (
    <>
      {/* Tuning trial chart — live series while running, persisted trials after.
    Boosting tuning jobs have both series; tabs switch between them. */}
      {hasBothSeries && (
        <div className="flex gap-2">
          {(['trial', 'iteration'] as SeriesKind[]).map((kind) => (
            <button
              key={kind}
              type="button"
              onClick={() => setPinnedSeries(kind)}
              aria-pressed={visibleKind === kind}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all whitespace-nowrap ${visibleKind === kind
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-700'
                }`}
            >
              {kind === 'trial' ? 'Trials' : 'Iterations'}
            </button>
          ))}
        </div>
      )}
      <TuningTrialsChart
        points={visibleSlice.points}
        metric={visibleSlice.metric ?? getScoringMetric(job)}
        isLive={!isTerminalStatus(job.status) && visibleSlice.points.length > 0}
        kind={visibleKind}
      />
    </>
  );
}
