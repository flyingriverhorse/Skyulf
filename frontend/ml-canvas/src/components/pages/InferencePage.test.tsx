import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { InferencePage, useSavedThresholdInfo } from './InferencePage';
import { thresholdTuningApi } from '../../core/api/thresholdTuning';
import { deploymentApi } from '../../core/api/deployment';
import { jobsApi } from '../../core/api/jobs';
import { DatasetService } from '../../core/api/datasets';

vi.mock('../../core/api/thresholdTuning', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/thresholdTuning')>(
    '../../core/api/thresholdTuning',
  );
  return {
    ...actual,
    thresholdTuningApi: {
      ...actual.thresholdTuningApi,
      get: vi.fn(),
    },
  };
});

vi.mock('../../core/api/deployment', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/deployment')>('../../core/api/deployment');
  return {
    ...actual,
    deploymentApi: {
      ...actual.deploymentApi,
      getActive: vi.fn(),
      predict: vi.fn(),
    },
  };
});

vi.mock('../../core/api/jobs', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/jobs')>('../../core/api/jobs');
  return {
    ...actual,
    jobsApi: {
      ...actual.jobsApi,
      getJob: vi.fn(),
    },
  };
});

vi.mock('../../core/api/datasets', async () => {
  const actual = await vi.importActual<typeof import('../../core/api/datasets')>('../../core/api/datasets');
  return {
    ...actual,
    DatasetService: {
      ...actual.DatasetService,
      getSample: vi.fn(),
    },
  };
});

vi.mock('../shared', () => ({
  useConfirm: () => vi.fn(async () => true),
}));

vi.mock('../../core/toast', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

const mockedThresholdTuningApi = vi.mocked(thresholdTuningApi);
const mockedDeploymentApi = vi.mocked(deploymentApi);
const mockedJobsApi = vi.mocked(jobsApi);
const mockedDatasetService = vi.mocked(DatasetService);

const activeDeployment = {
  id: 1,
  job_id: 'job-2',
  model_type: 'xgboost',
  artifact_uri: 's3://models/job-2',
  is_active: true,
  created_at: '2026-08-07T00:00:00.000Z',
};

const job = {
  job_id: 'job-2',
  pipeline_id: 'pipe-2',
  node_id: 'node-2',
  job_type: 'training' as const,
  status: 'completed' as const,
  start_time: null,
  end_time: null,
  error: null,
  result: null,
  created_at: '2026-08-07T00:00:00.000Z',
  target_column: 'target',
  dropped_columns: [],
  dataset_id: 'dataset-2',
};

beforeEach(() => {
  vi.clearAllMocks();
  mockedDeploymentApi.getActive.mockResolvedValue(activeDeployment);
  mockedJobsApi.getJob.mockResolvedValue(job as never);
  mockedDatasetService.getSample.mockResolvedValue([{ feature1: 0.2, feature2: 0.8 }] as never);
});

describe('useSavedThresholdInfo', () => {
  it('keeps saved thresholds tied to the currently selected job', async () => {
    let resolveJob1: ((value: Awaited<ReturnType<typeof thresholdTuningApi.get>>) => void) | undefined;
    let resolveJob2: ((value: Awaited<ReturnType<typeof thresholdTuningApi.get>>) => void) | undefined;

    mockedThresholdTuningApi.get
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveJob1 = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise(resolve => {
            resolveJob2 = resolve;
          }),
      );

    const Harness: React.FC<{ jobId: string | null }> = ({ jobId }) => {
      const savedThresholds = useSavedThresholdInfo(jobId);
      return (
        <div data-testid="saved-thresholds">
          {savedThresholds ? `${savedThresholds.metric}:${savedThresholds.split_used}` : 'none'}
        </div>
      );
    };

    const { rerender } = render(<Harness jobId="job-1" />);
    expect(screen.getByTestId('saved-thresholds')).toHaveTextContent('none');

    rerender(<Harness jobId="job-2" />);

    await waitFor(() => {
      expect(mockedThresholdTuningApi.get).toHaveBeenCalledTimes(2);
    });

    if (resolveJob1) {
      resolveJob1({
        thresholds: { a: 1 },
        classes: [0],
        metric: 'precision',
        split_used: 'test',
        computed_at: null,
        enabled: true,
      });
    }
    if (resolveJob2) {
      resolveJob2({
        thresholds: { a: 0.6 },
        classes: [0],
        metric: 'recall',
        split_used: 'validation',
        computed_at: null,
        enabled: true,
      });
    }

    await waitFor(() => {
      expect(screen.getByTestId('saved-thresholds')).toHaveTextContent('recall:validation');
    });
    expect(screen.getByTestId('saved-thresholds')).not.toHaveTextContent('precision:test');
    expect(mockedThresholdTuningApi.get).toHaveBeenCalledWith('job-1');
    expect(mockedThresholdTuningApi.get).toHaveBeenCalledWith('job-2');
  });
});

describe('InferencePage saved-threshold provenance', () => {
  it('shows the saved threshold job, model, metric, split, and saved values', async () => {
    mockedThresholdTuningApi.get.mockResolvedValue({
      thresholds: { '0': 0.7, '1': 0.4 },
      classes: [0, 1],
      metric: 'f1',
      split_used: 'validation',
      computed_at: '2026-08-07T12:34:56Z',
      enabled: true,
    } as never);

    render(
      <MemoryRouter>
        <InferencePage />
      </MemoryRouter>,
    );

    await screen.findByText('Advanced: override thresholds');
    fireEvent.click(screen.getByText('Advanced: override thresholds'));

    expect(await screen.findByText(/Job job-2 · xgboost/)).toBeInTheDocument();
    expect(
      screen.getAllByText((_, node) => node?.textContent?.includes('Optimized for f1') ?? false),
    ).not.toHaveLength(0);
    expect(
      screen.getAllByText((_, node) => node?.textContent?.includes('Computed from validation split') ?? false),
    ).not.toHaveLength(0);
    expect(
      screen.getAllByText((_, node) => node?.textContent?.includes('Computed at') ?? false),
    ).not.toHaveLength(0);
    expect(screen.getByText('0: 0.7')).toBeInTheDocument();
    expect(screen.getByText('1: 0.4')).toBeInTheDocument();
  });
});
