import { apiClient } from './client';

export interface ThresholdPreviewResult {
  thresholds: Record<string, number>;
  classes: number[];
  metric: string;
  split_used: string;
  /** Provenance, present only when this slot was hydrated from the job's
   * saved thresholds (e.g. `'training'` when they were seeded at training
   * time) — a live preview from the optimizer never carries one. */
  source?: string | null;
}

export interface ThresholdSavePayload {
  thresholds: Record<string, number>;
  classes: number[];
  metric: string;
  split_used: string;
}

/** A job's currently saved tuned thresholds, if any (`thresholds` etc. are
 * `null` and `enabled` is `false` when nothing has been saved yet). */
export interface SavedThresholdInfo {
  thresholds: Record<string, number> | null;
  classes: number[] | null;
  metric: string | null;
  split_used: string | null;
  computed_at: string | null;
  /** Where the saved set came from: `'training'` when seeded by
   * training-time threshold tuning, `null` for manually saved/legacy sets. */
  source: string | null;
  enabled: boolean;
}

export const thresholdTuningApi = {
  /** Fetch the job's currently saved tuned thresholds (if any) and enabled flag. */
  get: async (jobId: string): Promise<SavedThresholdInfo> => {
    const response = await apiClient.get<SavedThresholdInfo>(
      `/pipeline/jobs/${jobId}/thresholds`,
    );
    return response.data;
  },

  /** Compute (without saving) tuned per-class thresholds for a job's evaluation data. */
  preview: async (jobId: string, metric: string): Promise<ThresholdPreviewResult> => {
    const response = await apiClient.post<ThresholdPreviewResult>(
      `/pipeline/jobs/${jobId}/thresholds/preview`,
      { metric },
    );
    return response.data;
  },

  /** Persist tuned thresholds against the job. */
  save: async (jobId: string, payload: ThresholdSavePayload): Promise<void> => {
    await apiClient.post(`/pipeline/jobs/${jobId}/thresholds/save`, payload);
  },

  /** Enable or disable use of the job's saved tuned thresholds at predict time. */
  toggle: async (jobId: string, enabled: boolean): Promise<void> => {
    await apiClient.post(`/pipeline/jobs/${jobId}/thresholds/toggle`, { enabled });
  },

  /** Remove any saved tuned thresholds from the job. */
  clear: async (jobId: string): Promise<void> => {
    await apiClient.delete(`/pipeline/jobs/${jobId}/thresholds`);
  },
};
