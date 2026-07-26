import { describe, it, expect, vi, beforeEach } from 'vitest';
import { thresholdTuningApi } from './thresholdTuning';
import { apiClient } from './client';

// We mock the underlying HTTP layer (apiClient = axios instance) and assert
// the thresholdTuningApi wrapper sends the right URL/method/payload and
// reshapes responses correctly, matching the pattern used by jobs.test.ts.

describe('thresholdTuningApi.preview', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('posts metric and returns parsed thresholds', async () => {
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({
      data: {
        thresholds: { '0': 0.6, '1': 0.5, '2': 0.3 },
        classes: [0, 1, 2],
        metric: 'f1',
        split_used: 'validation',
      },
    } as unknown as Awaited<ReturnType<typeof apiClient.post>>);

    const result = await thresholdTuningApi.preview('job-1', 'f1');

    expect(post).toHaveBeenCalledWith('/pipeline/jobs/job-1/thresholds/preview', { metric: 'f1' });
    expect(result.thresholds).toEqual({ '0': 0.6, '1': 0.5, '2': 0.3 });
    expect(result.split_used).toBe('validation');
  });
});

describe('thresholdTuningApi.save', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs the full payload to /thresholds/save', async () => {
    const post = vi
      .spyOn(apiClient, 'post')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.post>>);

    const payload = {
      thresholds: { '0': 0.6 },
      classes: [0, 1],
      metric: 'f1',
      split_used: 'validation',
    };
    await thresholdTuningApi.save('job-1', payload);

    expect(post).toHaveBeenCalledWith('/pipeline/jobs/job-1/thresholds/save', payload);
  });
});

describe('thresholdTuningApi.toggle', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs enabled flag to /thresholds/toggle', async () => {
    const post = vi
      .spyOn(apiClient, 'post')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.post>>);

    await thresholdTuningApi.toggle('job-1', true);

    expect(post).toHaveBeenCalledWith('/pipeline/jobs/job-1/thresholds/toggle', { enabled: true });
  });
});

describe('thresholdTuningApi.clear', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('DELETEs /thresholds', async () => {
    const del = vi
      .spyOn(apiClient, 'delete')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.delete>>);

    await thresholdTuningApi.clear('job-1');

    expect(del).toHaveBeenCalledWith('/pipeline/jobs/job-1/thresholds');
  });
});
