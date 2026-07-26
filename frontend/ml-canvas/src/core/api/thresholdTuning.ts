import { apiClient } from './client';

export interface ThresholdPreviewResult {
  thresholds: Record<string, number>;
  classes: number[];
  metric: string;
  split_used: string;
}

export interface ThresholdSavePayload {
  thresholds: Record<string, number>;
  classes: number[];
  metric: string;
  split_used: string;
}

export const thresholdTuningApi = {
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
