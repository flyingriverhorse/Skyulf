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

vi.mock('../shared', async () => {
  const actual = await vi.importActual<typeof import('../shared')>('../shared');
  return {
    ...actual,
    useConfirm: () => vi.fn(async () => true),
  };
});

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
  localStorage.clear();
  mockedDeploymentApi.getActive.mockResolvedValue(activeDeployment);
  mockedJobsApi.getJob.mockResolvedValue(job as never);
  mockedDatasetService.getSample.mockResolvedValue([{ feature1: 0.2, feature2: 0.8 }] as never);
  mockedThresholdTuningApi.get.mockResolvedValue(null as never);
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
        source: null,
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
        source: null,
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
      source: 'training',
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
      screen.getAllByText((_, node) => node?.textContent?.includes('Seeded at training time') ?? false),
    ).not.toHaveLength(0);
    expect(
      screen.getAllByText((_, node) => node?.textContent?.includes('Computed at') ?? false),
    ).not.toHaveLength(0);
    expect(screen.getByText('0: 0.7')).toBeInTheDocument();
    expect(screen.getByText('1: 0.4')).toBeInTheDocument();
  });
});

/** Build a `deploymentApi.predict` mock that resolves/rejects only when the
 * test calls the returned control functions, and rejects with an
 * axios-style cancellation error when the request's AbortSignal fires. */
function deferredPredict() {
  let resolveFn: ((value: Awaited<ReturnType<typeof deploymentApi.predict>>) => void) | undefined;
  let rejectFn: ((reason: unknown) => void) | undefined;
  const impl = vi.fn(
    (
      _data: unknown[],
      _overrideThresholds?: Record<string, number> | null,
      options?: { signal?: AbortSignal },
    ) =>
      new Promise<Awaited<ReturnType<typeof deploymentApi.predict>>>((resolve, reject) => {
        resolveFn = resolve;
        rejectFn = reject;
        options?.signal?.addEventListener('abort', () => {
          const err = new Error('canceled') as Error & { code: string };
          err.code = 'ERR_CANCELED';
          reject(err);
        });
      }),
  );
  return {
    impl,
    resolve: (value: Awaited<ReturnType<typeof deploymentApi.predict>>) => resolveFn?.(value),
    reject: (reason: unknown) => rejectFn?.(reason),
  };
}

const renderInferencePage = () =>
  render(
    <MemoryRouter>
      <InferencePage />
    </MemoryRouter>,
  );

describe('InferencePage run lifecycle (EXP-007)', () => {
  it('names the pending run, disables Run Prediction, and prevents a duplicate submission', async () => {
    const { impl, resolve } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    fireEvent.click(runButton);

    expect(await screen.findByText(/Run #1 is running/i)).toBeInTheDocument();
    const pendingButton = screen.getByRole('button', { name: /Run #1 running/i });
    expect(pendingButton).toBeDisabled();

    // A second click while pending must not submit a second request.
    fireEvent.click(pendingButton);
    expect(impl).toHaveBeenCalledTimes(1);

    resolve({ predictions: [1], model_version: 'job-2' });
    await waitFor(() => expect(screen.queryByText(/is running/i)).not.toBeInTheDocument());
  });

  it('shows a safe cause, keeps input unchanged, and retries in place on failure', async () => {
    const { impl, reject } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    const inputBefore = (screen.getByLabelText(/Input Data/i) as HTMLTextAreaElement)?.value ??
      (document.getElementById('inference-input-editor') as HTMLTextAreaElement).value;
    fireEvent.click(runButton);

    reject(new Error('Feature engineering failed: bad dtype'));

    expect(await screen.findByRole('alert')).toHaveTextContent('Feature engineering failed: bad dtype');
    const editor = document.getElementById('inference-input-editor') as HTMLTextAreaElement;
    expect(editor.value).toBe(inputBefore);

    // Retry resends the same request as a new named run.
    const { impl: retryImpl, resolve: retryResolve } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(retryImpl);
    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    expect(await screen.findByText(/Run #2.*is running/i)).toBeInTheDocument();
    retryResolve({ predictions: [1], model_version: 'job-2' });
    await waitFor(() => expect(screen.queryByText(/is running/i)).not.toBeInTheDocument());
    expect(retryImpl).toHaveBeenCalledTimes(1);
  });

  it('cancels a pending run in place and lets the user retry the same input', async () => {
    const { impl } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    fireEvent.click(runButton);
    await screen.findByText(/Run #1 is running/i);

    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }));

    expect(await screen.findByText(/Run cancelled/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Retry same input/i })).toBeInTheDocument();
  });

  it('shows result/model/threshold/row-count provenance on success and survives reload via history', async () => {
    const { impl, resolve } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    const { unmount } = renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    fireEvent.click(runButton);
    resolve({ predictions: [1], model_version: 'v-42' });

    const provenance = await screen.findByTestId('run-provenance');
    expect(provenance).toHaveTextContent('Run #1');
    expect(provenance).toHaveTextContent('xgboost');
    expect(provenance).toHaveTextContent('v-42');
    expect(provenance).toHaveTextContent('job-2');
    expect(provenance).toHaveTextContent('1 row');

    unmount();

    // Reload: a fresh mount hydrates the last settled run from durable history.
    renderInferencePage();
    const restoredProvenance = await screen.findByTestId('run-provenance');
    expect(restoredProvenance).toHaveTextContent('Run #1');
    expect(restoredProvenance).toHaveTextContent('v-42');
    expect(screen.getAllByText(/Recent runs/i).length).toBeGreaterThan(0);
  });

  it('marks a run interrupted by reload as failed with an explicit, non-raw cause', async () => {
    const { impl } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    const { unmount } = renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    fireEvent.click(runButton);
    await screen.findByText(/Run #1 is running/i);

    // Simulate the tab closing/reloading mid-flight: unmount without settling.
    unmount();

    renderInferencePage();
    expect(await screen.findByRole('alert')).toHaveTextContent(/reloaded or closed/i);
  });

  it('names the export filename with job and run identifiers', async () => {
    const { impl, resolve } = deferredPredict();
    mockedDeploymentApi.predict.mockImplementation(impl);

    renderInferencePage();
    const runButton = await screen.findByRole('button', { name: /Run Prediction/i });
    fireEvent.click(runButton);
    resolve({ predictions: [1], model_version: 'v-42' });
    await screen.findByTestId('run-provenance');

    const clickSpy = vi.fn();
    const originalCreateElement = document.createElement.bind(document);
    const createElementSpy = vi
      .spyOn(document, 'createElement')
      .mockImplementation((tag: string) => {
        const el = originalCreateElement(tag);
        if (tag === 'a') el.click = clickSpy;
        return el;
      });
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:mock');
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);

    fireEvent.click(screen.getByRole('button', { name: /^JSON$/i }));

    expect(clickSpy).toHaveBeenCalledTimes(1);
    const anchor = createElementSpy.mock.results.find(r => (r.value as HTMLElement).tagName === 'A')
      ?.value as HTMLAnchorElement;
    expect(anchor.download).toContain('job-2');

    createElementSpy.mockRestore();
    createObjectURLSpy.mockRestore();
    revokeSpy.mockRestore();
  });
});
