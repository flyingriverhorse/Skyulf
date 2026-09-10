import { useMemo } from 'react';
import { diffGraphs, type GraphDiff } from '../../../../core/utils/graphDiff';
import { applyDiffStylingToSide, applyLayout, layoutUnified, type JobLite } from '../pipelineDiffLayout';
import type { SnapshotState } from './usePipelineSnapshots';

export function usePipelineComparison(
  selectedJobs: JobLite[],
  swapped: boolean,
  snapshots: Record<string, SnapshotState>,
) {
  const baselineJob = selectedJobs.length === 2 ? (swapped ? selectedJobs[1] : selectedJobs[0]) : undefined;
  const candidateJob = selectedJobs.length === 2 ? (swapped ? selectedJobs[0] : selectedJobs[1]) : undefined;

  const diff = useMemo<GraphDiff | null>(() => {
    const baseline = baselineJob ? snapshots[baselineJob.job_id] : undefined;
    const candidate = candidateJob ? snapshots[candidateJob.job_id] : undefined;
    if (baseline?.status !== 'ready' || candidate?.status !== 'ready') return null;
    return diffGraphs(baseline.graph.nodes, baseline.graph.edges, candidate.graph.nodes, candidate.graph.edges);
  }, [baselineJob, candidateJob, snapshots]);

  const styled = useMemo(() => {
    const baseline = baselineJob ? snapshots[baselineJob.job_id] : undefined;
    const candidate = candidateJob ? snapshots[candidateJob.job_id] : undefined;
    if (baseline?.status !== 'ready' || candidate?.status !== 'ready' || !diff) return null;
    const { positions } = layoutUnified(baseline.graph, candidate.graph, diff.aliases);
    return {
      baseline: applyLayout(applyDiffStylingToSide(baseline.graph, diff, 'left'), positions, diff.aliases),
      candidate: applyLayout(applyDiffStylingToSide(candidate.graph, diff, 'right'), positions, diff.aliases),
    };
  }, [baselineJob, candidateJob, snapshots, diff]);

  return { baselineJob, candidateJob, diff, styled };
}
