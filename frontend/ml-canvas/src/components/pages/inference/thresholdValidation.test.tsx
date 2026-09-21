import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../../../core/api/client';
import { deploymentApi } from '../../../core/api/deployment';
import { jobsApi } from '../../../core/api/jobs';
import { thresholdTuningApi } from '../../../core/api/thresholdTuning';
import { ConfirmProvider } from '../../shared/ConfirmDialog';
import { LS_INPUT } from './inferenceData';
import { ThresholdOverrides } from './ThresholdOverrides';
import { useInferenceController } from './useInferenceController';

describe('inference threshold validation feedback', () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem(LS_INPUT, '[{"a":1}]');
    vi.spyOn(deploymentApi, 'getActive').mockResolvedValue({
      id: 1, job_id: 'job-a', model_type: 'classifier', artifact_uri: 'model.joblib',
      is_active: true, created_at: '2026-09-12T00:00:00Z', input_schema: [{ name: 'a', type: 'float64' }],
    });
    vi.spyOn(jobsApi, 'getJob').mockResolvedValue({
      job_id: 'job-a', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
      status: 'completed', start_time: null, end_time: null, error: null, result: null,
      created_at: '2026-09-12T00:00:00Z',
    });
    vi.spyOn(thresholdTuningApi, 'get').mockResolvedValue({
      thresholds: { '0': 2, '1': 3, '2': 4 }, classes: [0, 1, 2], metric: 'f1', split_used: 'validation',
      computed_at: null, source: null, enabled: false,
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it('allows above-one multiclass values and reports an invalid zero without submitting', async () => {
    /** The editor and failed prediction feedback must follow the real client denominator contract. */
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} });
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.savedThresholds?.classes).toHaveLength(3));
    act(() => result.current.handlePrefillFromSavedThresholds());
    render(<ThresholdOverrides controller={result.current} />);
    expect(screen.getAllByRole('spinbutton', { hidden: true })[0]).not.toHaveAttribute('max');
    act(() => result.current.handleOverrideThresholdChange('1', '0'));
    await act(() => result.current.handlePredict());
    expect(result.current.error).toMatch(/finite.*greater than 0/i);
    expect(result.current.predictions).toBeNull();
    expect(post).not.toHaveBeenCalled();
  });

  it('reports blank or non-finite edits instead of silently keeping the previous value', async () => {
    /** Users must know that a malformed edit was rejected before running a prediction. */
    const { result } = renderHook(useInferenceController, { wrapper: ConfirmProvider });
    await waitFor(() => expect(result.current.savedThresholds?.classes).toHaveLength(3));
    act(() => result.current.handlePrefillFromSavedThresholds());
    for (const value of ['', 'Infinity', 'NaN']) {
      act(() => result.current.handleOverrideThresholdChange('1', value));
      expect(result.current.error).toMatch(/finite number/i);
      expect(result.current.overrideThresholdsValue['1']).toBe(3);
    }
  });
});
