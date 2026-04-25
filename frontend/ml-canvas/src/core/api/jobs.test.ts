import { describe, it, expect, vi, beforeEach } from 'vitest';
import { jobsApi } from './jobs';
import { apiClient } from './client';
import axios from 'axios';

// We mock the underlying HTTP layer (apiClient = axios instance, plus
// the bare `axios` import used by getIngestionJobs) and assert the
// jobsApi wrapper sends the right URL/method/payload and reshapes
// responses correctly.

describe('jobsApi.runPipeline', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs to /pipeline/run and returns the response data', async () => {
    const post = vi.spyOn(apiClient, 'post').mockResolvedValue({
      data: { message: 'ok', pipeline_id: 'p1', job_id: 'j1', job_ids: ['j1'] },
    } as unknown as Awaited<ReturnType<typeof apiClient.post>>);
    const result = await jobsApi.runPipeline({ nodes: [], edges: [] } as never);
    expect(post).toHaveBeenCalledWith('/pipeline/run', { nodes: [], edges: [] });
    expect(result.pipeline_id).toBe('p1');
  });
});

describe('jobsApi.getJob / cancelJob / promote / unpromote', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('GETs /pipeline/jobs/:id', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: { job_id: 'j1' } } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    const job = await jobsApi.getJob('j1');
    expect(get).toHaveBeenCalledWith('/pipeline/jobs/j1');
    expect(job.job_id).toBe('j1');
  });

  it('cancelJob POSTs /pipeline/jobs/:id/cancel', async () => {
    const post = vi
      .spyOn(apiClient, 'post')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.post>>);
    await jobsApi.cancelJob('j2');
    expect(post).toHaveBeenCalledWith('/pipeline/jobs/j2/cancel');
  });

  it('promoteJob POSTs /pipeline/jobs/:id/promote', async () => {
    const post = vi
      .spyOn(apiClient, 'post')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.post>>);
    await jobsApi.promoteJob('j3');
    expect(post).toHaveBeenCalledWith('/pipeline/jobs/j3/promote');
  });

  it('unpromoteJob DELETEs /pipeline/jobs/:id/promote', async () => {
    const del = vi
      .spyOn(apiClient, 'delete')
      .mockResolvedValue({ data: undefined } as unknown as Awaited<ReturnType<typeof apiClient.delete>>);
    await jobsApi.unpromoteJob('j4');
    expect(del).toHaveBeenCalledWith('/pipeline/jobs/j4/promote');
  });
});

describe('jobsApi.getJobs', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('forwards limit/skip and the optional job_type param', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: [] } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    await jobsApi.getJobs(20, 5, 'tuning');
    expect(get).toHaveBeenCalledWith('/pipeline/jobs', {
      params: { limit: 20, skip: 5, job_type: 'tuning' },
    });
  });

  it('omits job_type when not provided', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: [] } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    await jobsApi.getJobs();
    const call = get.mock.calls[0]!;
    expect(call[0]).toBe('/pipeline/jobs');
    expect((call[1] as { params: { job_type?: unknown } }).params.job_type).toBeUndefined();
  });
});

describe('jobsApi.getEDAJobs', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('shapes raw EDA job records into JobInfo objects', async () => {
    vi.spyOn(apiClient, 'get').mockResolvedValue({
      data: [
        {
          id: 42,
          dataset_id: 7,
          dataset_name: 'iris',
          status: 'COMPLETED',
          created_at: '2026-04-25T10:00:00Z',
          updated_at: '2026-04-25T10:01:00Z',
          error: null,
          target_col: 'species',
        },
      ],
    } as unknown as Awaited<ReturnType<typeof apiClient.get>>);

    const jobs = await jobsApi.getEDAJobs(50);
    expect(jobs).toHaveLength(1);
    expect(jobs[0]).toMatchObject({
      job_id: '42',
      dataset_id: '7',
      dataset_name: 'iris',
      job_type: 'eda',
      // Status is lowercased by the wrapper.
      status: 'completed',
      target_column: 'species',
    });
  });
});

describe('jobsApi.getIngestionJobs', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('hits /data/api/sources directly (bypasses the /api prefix) and maps test_status', async () => {
    const get = vi.spyOn(axios, 'get').mockResolvedValue({
      data: {
        sources: [
          {
            id: 1,
            name: 'csv-1',
            type: 'csv',
            test_status: 'success',
            created_at: '2026-04-25T09:00:00Z',
            updated_at: '2026-04-25T09:00:00Z',
          },
          {
            id: 2,
            name: 'pg-1',
            type: 'postgres',
            test_status: 'failed',
            created_at: '2026-04-25T08:00:00Z',
            updated_at: '2026-04-25T08:00:00Z',
          },
        ],
      },
    } as unknown as Awaited<ReturnType<typeof axios.get>>);

    const jobs = await jobsApi.getIngestionJobs(50, 0);
    expect(get).toHaveBeenCalledWith('/data/api/sources', { params: { limit: 50, skip: 0 } });
    expect(jobs).toHaveLength(2);
    expect(jobs[0]?.status).toBe('succeeded');
    expect(jobs[1]?.status).toBe('failed');
    expect(jobs[0]?.model_type).toBe('csv');
  });
});

describe('jobsApi miscellaneous', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('getHyperparameters hits /pipeline/hyperparameters/:type', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: [] } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    await jobsApi.getHyperparameters('xgboost');
    expect(get).toHaveBeenCalledWith('/pipeline/hyperparameters/xgboost');
  });

  it('getDefaultSearchSpace hits /pipeline/hyperparameters/:type/defaults', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: {} } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    await jobsApi.getDefaultSearchSpace('rf');
    expect(get).toHaveBeenCalledWith('/pipeline/hyperparameters/rf/defaults');
  });

  it('getLatestTuningJob / getBestTuningJob / getTuningHistory hit the right paths', async () => {
    const get = vi
      .spyOn(apiClient, 'get')
      .mockResolvedValue({ data: null } as unknown as Awaited<ReturnType<typeof apiClient.get>>);
    await jobsApi.getLatestTuningJob('node-1');
    await jobsApi.getBestTuningJob('xgboost');
    await jobsApi.getTuningHistory('rf');
    const urls = get.mock.calls.map((c) => c[0]);
    expect(urls).toEqual([
      '/pipeline/jobs/tuning/latest/node-1',
      '/pipeline/jobs/tuning/best/xgboost',
      '/pipeline/jobs/tuning/history/rf',
    ]);
  });
});
