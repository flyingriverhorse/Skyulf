/**
 * Evaluation fetch + threshold-tuning state for the Experiments page.
 *
 * Extracted from ExperimentsPage.tsx so the response-ordering guard
 * (a late response for job A must not clobber data already rendered for
 * job B) can be unit-tested in isolation.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '../../../../core/api/client';
import { thresholdTuningApi, type ThresholdPreviewResult } from '../../../../core/api/thresholdTuning';
import type { JobInfo } from '../../../../core/api/jobs';
import type { EvaluationData } from '../types';
import { getJobScoringMetric, mapJobMetricToDropdown, type ThresholdMetric } from '../utils/jobMeta';

/**
 * Owns the evaluation data load for the active job plus the
 * threshold-tuning panel state that is reset/hydrated alongside it.
 *
 * Responses are guarded by a monotonic request sequence: when the user
 * clicks job A (slow) and then job B (fast), A's late response is
 * discarded instead of being rendered under B's header.
 */
export function useEvaluationFetch(jobs: JobInfo[]) {
  const [evaluationData, setEvaluationData] = useState<EvaluationData | null>(null);
  const [isEvalLoading, setIsEvalLoading] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [evalJobId, setEvalJobId] = useState<string | null>(null);
  const [selectedTuningMetric, setSelectedTuningMetric] = useState<string>('f1');
  const [tuningPreview, setTuningPreview] = useState<ThresholdPreviewResult | null>(null);
  const [useTunedThresholds, setUseTunedThresholds] = useState(false);
  const [tuningError, setTuningError] = useState<string | null>(null);
  // Which metric the classification best-threshold scan optimizes for.
  // Reset per-job in fetchEvaluationData below, defaulting to the job's
  // own scoring metric (via mapJobMetricToDropdown) instead of always F1.
  const [selectedThresholdMetric, setSelectedThresholdMetric] = useState<ThresholdMetric>('f1_weighted');

  // Ref mirror of `jobs`, read (not depended on) inside fetchEvaluationData
  // so looking up the job's own scoring metric doesn't force that callback
  // to be treated as reactive on every jobs-list refresh/poll.
  const jobsRef = useRef(jobs);
  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  // Monotonic sequence: every fetchEvaluationData call claims the next
  // number; a response may only touch state while its number is still
  // the most recent one (i.e. no newer fetch was started).
  const requestSeq = useRef(0);

  const fetchEvaluationData = useCallback(async (jobId: string) => {
    const seq = ++requestSeq.current;
    const isStale = () => seq !== requestSeq.current;

    // Stale-while-revalidate: keep showing the previously rendered
    // charts while the new run loads. Setting `evaluationData` to
    // null here would unmount the entire panel and flash the
    // spinner on every job switch — the "blink" the user reported
    // when clicking between runs in the Model Evaluation tab.
    setIsEvalLoading(true);
    setEvalError(null);
    setEvalJobId(jobId);
    // Default the metric dropdown to this job's own scoring metric (not
    // always F1) — the user can still change it afterward for this job.
    const job = jobsRef.current.find(j => j.job_id === jobId);
    setSelectedThresholdMetric(mapJobMetricToDropdown(job ? getJobScoringMetric(job) : undefined));
    // Reset threshold-tuning UI state — a preview/tuned-thresholds state
    // from a previously viewed job must not leak onto the newly selected
    // one (they're keyed per-job server-side too).
    setTuningPreview(null);
    setUseTunedThresholds(false);
    setTuningError(null);
    try {
      const res = await apiClient.get(`/pipeline/jobs/${jobId}/evaluation`);
      if (isStale()) return;
      setEvaluationData(res.data);
    } catch (err: unknown) {
      if (isStale()) return;
      console.error('Failed to fetch evaluation data', err);
      setEvalError((err as { response?: { data?: { detail?: string } } }).response?.data?.detail || 'Failed to fetch evaluation data');
      setEvaluationData(null);
    } finally {
      if (!isStale()) {
        setIsEvalLoading(false);
      }
    }
    // Hydrate the Tuning tab from whatever this job already has saved
    // server-side, instead of always starting unchecked/empty — without
    // this, reopening a job that already has tuned thresholds saved and
    // enabled would silently show the toggle off even though real
    // /predict calls for it are already using those thresholds.
    try {
      const saved = await thresholdTuningApi.get(jobId);
      if (isStale()) return;
      if (saved.thresholds && saved.classes && saved.metric && saved.split_used) {
        setTuningPreview({
          thresholds: saved.thresholds,
          classes: saved.classes,
          metric: saved.metric,
          split_used: saved.split_used,
        });
        setSelectedTuningMetric(saved.metric);
        setUseTunedThresholds(saved.enabled);
      }
    } catch (err: unknown) {
      // Non-fatal — the Tuning tab just starts from its reset defaults.
      if (!isStale()) {
        console.error('Failed to fetch saved tuned thresholds', err);
      }
    }
  }, []);

  return {
    evaluationData,
    setEvaluationData,
    isEvalLoading,
    evalError,
    setEvalError,
    evalJobId,
    setEvalJobId,
    selectedTuningMetric,
    setSelectedTuningMetric,
    tuningPreview,
    setTuningPreview,
    useTunedThresholds,
    setUseTunedThresholds,
    tuningError,
    setTuningError,
    selectedThresholdMetric,
    setSelectedThresholdMetric,
    fetchEvaluationData,
  };
}
