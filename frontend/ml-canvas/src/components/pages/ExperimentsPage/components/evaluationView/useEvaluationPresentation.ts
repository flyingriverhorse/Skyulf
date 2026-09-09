import { useMemo } from 'react';
import { shortRunId } from '../../utils/jobMeta';
import type { EvaluationViewProps } from './types';

/** Share the same available split and default between controls and rendered charts. */
export function useEvaluationPresentation({ evaluationData, selectedRegressionSplit, eligibleJobs, eligibleJobIds }: EvaluationViewProps) {
  // Tabs and charts share this derivation. The backend split key is
  // 'validation'; 'val_' is only the flattened metric prefix.
  const availableRegressionSplits = useMemo(() => {
    if (!evaluationData) return [];
    return (Object.keys(evaluationData.splits) as string[]).filter(
      (s) => evaluationData.splits[s as keyof typeof evaluationData.splits] != null
    );
  }, [evaluationData]);

  const regressionSplitTabs = useMemo(
    () => ['train', 'test', 'validation'].filter((s) => availableRegressionSplits.includes(s)),
    [availableRegressionSplits]
  );

  const regressionSplitLabels: Record<string, string> = {
    train: 'Train',
    test: 'Test',
    validation: 'Validation',
  };

  // Classification split-toggle availability — hides a Train/Test/Validation
  // checkbox entirely when this job has no data for that split at all,
  // instead of showing a checkbox that toggles nothing.
  const hasTrainSplit = !!evaluationData?.splits.train;
  const hasTestSplit = !!evaluationData?.splits.test;
  const hasValidationSplit = !!evaluationData?.splits.validation;

  const activeRegressionSplit = useMemo(() => {
    if (selectedRegressionSplit != null && availableRegressionSplits.includes(selectedRegressionSplit)) {
      return selectedRegressionSplit;
    }
    return (
      regressionSplitTabs.find((t) => t === 'validation') ??
      regressionSplitTabs.find((t) => t === 'test') ??
      regressionSplitTabs[0] ??
      availableRegressionSplits[0]
    );
  }, [selectedRegressionSplit, availableRegressionSplits, regressionSplitTabs]);

  const eligibleRunLabels = useMemo(() => {
    if (eligibleJobs && eligibleJobs.length > 0) {
      return eligibleJobs.map((job) => ({ jobId: job.jobId, label: shortRunId(job) }));
    }
    return eligibleJobIds.map((jobId) => ({ jobId, label: `Job ID: ${jobId}` }));
  }, [eligibleJobs, eligibleJobIds]);

  return { regressionSplitTabs, regressionSplitLabels, activeRegressionSplit, eligibleRunLabels, hasTrainSplit, hasTestSplit, hasValidationSplit };
}

export type EvaluationPresentation = ReturnType<typeof useEvaluationPresentation>;
