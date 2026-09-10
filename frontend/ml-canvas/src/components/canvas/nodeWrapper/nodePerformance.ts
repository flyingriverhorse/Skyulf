import { useViewStore } from '../../../core/store/useViewStore';
import { bucketDuration, getPerfFamily } from '../../../core/perf/perfThresholds';
import type { NodeExecutionResult } from '../../../core/api/client';
import type { NodeSummaryEntry } from '../../../core/api/jobs';

/** Prefer preview timing, falling back to the first job that supplied duration. */
function getDuration(perfOverlayEnabled: boolean, nodeResult: NodeExecutionResult | undefined,
  jobSummaries: NodeSummaryEntry[] | undefined): number | null {
  if (!perfOverlayEnabled) return null;
  if (typeof nodeResult?.execution_time === 'number') {
    return Math.max(0, Math.round(nodeResult.execution_time * 1000));
  }
  const fromJob = jobSummaries?.find((e) => typeof e.duration_ms === 'number');
  if (fromJob && typeof fromJob.duration_ms === 'number') return fromJob.duration_ms;
  return null;
}

function metricDetails(m: NodeExecutionResult['metrics']) {
  let fitStr: string | null = null;
  let memMB: number | null = null;
  let rowsStr: string | null = null;
  if (m) {
    if (typeof m.fit_time === 'number') {
      fitStr = m.fit_time >= 1
        ? `${m.fit_time.toFixed(2)}s`
        : `${Math.round(m.fit_time * 1000)}ms`;
    }
    if (typeof m.peak_memory_bytes === 'number') {
      memMB = m.peak_memory_bytes / (1024 * 1024);
    }
    if (typeof m.rows_in === 'number' && typeof m.rows_out === 'number') {
      rowsStr = `${m.rows_in} \u2192 ${m.rows_out}`;
    }
  }
  return { fitStr, memMB, rowsStr };
}

function telemetry(perfDurationMs: number | null, nodeResult: NodeExecutionResult | undefined) {
  if (perfDurationMs === null) return null;

  // Core wall-clock duration message:
  const durStr =
    perfDurationMs >= 1000
      ? `${(perfDurationMs / 1000).toFixed(2)}s`
      : `${perfDurationMs}ms`;

  const { fitStr, memMB, rowsStr } = metricDetails(nodeResult?.metrics);
  let tooltip = `Last run: ${durStr}`;
  if (fitStr) tooltip += `\nFit time: ${fitStr}`;
  if (memMB !== null) tooltip += `\nPeak mem: ${memMB.toFixed(1)} MB`;
  if (rowsStr) tooltip += `\nRows: ${rowsStr}`;

  return { durStr, fitStr, memMB, rowsStr, tooltip };
}

/** Keep durations, color buckets, tooltip details, and footer data in agreement. */
export function useNodePerformance(definitionType: string, runMode: string | undefined,
  nodeResult: NodeExecutionResult | undefined, jobSummaries: NodeSummaryEntry[] | undefined) {
  const perfOverlayEnabled = useViewStore((s) => s.perfOverlayEnabled);
  const perfDurationMs = getDuration(perfOverlayEnabled, nodeResult, jobSummaries);
  const perfBucket: 'fast' | 'medium' | 'slow' | null =
    perfDurationMs === null
      ? null
      : bucketDuration(perfDurationMs, getPerfFamily(definitionType, runMode));
  const perfRingClass =
    perfBucket === 'fast'
      ? 'ring-2 ring-green-500/60 ring-offset-1 ring-offset-background'
      : perfBucket === 'medium'
        ? 'ring-2 ring-amber-500/70 ring-offset-1 ring-offset-background'
        : perfBucket === 'slow'
          ? 'ring-2 ring-red-500/70 ring-offset-1 ring-offset-background'
          : '';
  const perfTelemetry = telemetry(perfDurationMs, nodeResult);
  return { perfOverlayEnabled, perfDurationMs, perfBucket, perfRingClass, perfTelemetry };
}

export type NodePerformance = ReturnType<typeof useNodePerformance>;
