import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { JobDetailsView } from './JobDetailsView';
import { jobsApi, JobInfo } from '../../../core/api/jobs';
import { monitoringApi } from '../../../core/api/monitoring';

vi.mock('../../../core/api/monitoring', async () => {
  const actual = await vi.importActual<typeof import('../../../core/api/monitoring')>(
    '../../../core/api/monitoring',
  );
  return {
    ...actual,
    monitoringApi: {
      ...actual.monitoringApi,
      getJobNode: vi.fn(),
    },
  };
});

const mocks = vi.hoisted(() => ({
  cancelJob: vi.fn(),
  retryJob: vi.fn(),
  confirm: vi.fn(),
  toastSuccess: vi.fn(),
  toastError: vi.fn(),
}));

vi.mock('../../../core/store/useJobStore', () => ({
  useJobStore: () => ({ cancelJob: mocks.cancelJob, retryJob: mocks.retryJob }),
}));

vi.mock('../../../core/toast', () => ({
  toast: { success: mocks.toastSuccess, error: mocks.toastError },
}));

vi.mock('../../shared', async () => {
  const actual = await vi.importActual<typeof import('../../shared')>('../../shared');
  return { ...actual, useConfirm: () => mocks.confirm };
});

// jsdom has no layout engine for recharts' ResponsiveContainer — passthrough
// with explicit dimensions so the tuning trial chart renders in tests (same
// convention as FeatureImportanceView.test.tsx).
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <div style={{ width: 800, height: 240 }}>
        {React.isValidElement(children)
          ? React.cloneElement(children, { width: 800, height: 240 } as never)
          : children}
      </div>
    ),
  };
});

const makeJob = (overrides: Partial<JobInfo> = {}): JobInfo => ({
  job_id: 'job-1234567890',
  pipeline_id: 'pipe-1',
  node_id: 'node-1',
  job_type: 'training',
  status: 'failed',
  start_time: '2026-01-01T00:00:00Z',
  end_time: '2026-01-01T00:05:00Z',
  error: null,
  result: null,
  created_at: '2026-01-01T00:00:00Z',
  ...overrides,
});

function renderDetails(job: JobInfo, props: Partial<React.ComponentProps<typeof JobDetailsView>> = {}) {
  const onBack = vi.fn();
  const onClose = vi.fn();
  render(
    <MemoryRouter>
      <JobDetailsView job={job} onBack={onBack} onClose={onClose} {...props} />
    </MemoryRouter>,
  );
  return { onBack, onClose };
}

describe('JobDetailsView', () => {
  let originalScrollIntoView: typeof HTMLElement.prototype.scrollIntoView | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom has no layout engine and doesn't implement scrollIntoView; the
    // Logs tab's auto-scroll effect calls it on every render.
    originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    HTMLElement.prototype.scrollIntoView = vi.fn();
    // Non-terminal jobs feed their id into useJobPolling, which fetches
    // for real via jobsApi.getJob — stub it so in-flight jobs don't hit
    // the network (jsdom otherwise logs a noisy connection failure).
    vi.spyOn(jobsApi, 'getJob').mockImplementation((id: string) => Promise.resolve(makeJob({ job_id: id, status: 'running' })));
  });

  afterEach(() => {
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView ?? (() => {});
  });

  it('names the full job id for assistive tech even though the header truncates it', () => {
    renderDetails(makeJob({ job_id: 'job-abcdef123456' }));
    const idChip = screen.getByTitle('job-abcdef123456');
    // The full id is present in the DOM (sr-only), not just the 8-char prefix.
    expect(idChip).toHaveTextContent('job-abcdef123456');
  });

  it('shows an active Retry action for a failed, retryable training job', async () => {
    mocks.confirm.mockResolvedValue(true);
    mocks.retryJob.mockResolvedValue('job-new-1');
    const { onBack } = renderDetails(makeJob({ status: 'failed', job_type: 'training' }));

    const retryButton = screen.getByRole('button', { name: /^Retry$/ });
    expect(retryButton).toBeEnabled();

    await act(async () => {
      fireEvent.click(retryButton);
    });

    await waitFor(() => {
      expect(mocks.retryJob).toHaveBeenCalledWith('job-1234567890');
    });
    expect(onBack).toHaveBeenCalled();
  });

  it('never double-submits retry: a second click while the first is in flight is ignored', async () => {
    mocks.confirm.mockResolvedValue(true);
    let resolveRetry: (() => void) | undefined;
    mocks.retryJob.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveRetry = () => { resolve('job-new-1'); };
        }),
    );
    renderDetails(makeJob({ status: 'failed', job_type: 'training' }));

    const retryButton = screen.getByRole('button', { name: /^Retry$/ });
    await act(async () => {
      fireEvent.click(retryButton);
    });

    // Button now reads "Retrying..." and is disabled — a second click must not fire another call.
    const retryingButton = screen.getByRole('button', { name: /Retrying/ });
    expect(retryingButton).toBeDisabled();
    fireEvent.click(retryingButton);

    expect(mocks.retryJob).toHaveBeenCalledTimes(1);
    resolveRetry?.();
    await waitFor(() => expect(screen.getByRole('button', { name: /^Retry$/ })).toBeEnabled());
  });

  it('explains rather than hides retry when the job type does not support it', () => {
    renderDetails(makeJob({ status: 'failed', job_type: 'eda' }));
    expect(screen.queryByRole('button', { name: /^Retry$/ })).not.toBeInTheDocument();
    const unavailable = screen.getByText('Retry unavailable');
    expect(unavailable).toHaveAttribute('title', expect.stringContaining("isn't available for eda jobs"));
  });

  it('explains retry unavailability for a job that already succeeded', () => {
    renderDetails(makeJob({ status: 'completed', job_type: 'training' }));
    const unavailable = screen.getByText('Retry unavailable');
    expect(unavailable).toHaveAttribute('title', expect.stringContaining('completed successfully'));
  });

  it('shows a Stop action for a running job and guards it against double-submission', async () => {
    mocks.confirm.mockResolvedValue(true);
    let resolveCancel: (() => void) | undefined;
    mocks.cancelJob.mockImplementation(
      () =>
        new Promise(resolve => {
          resolveCancel = () => { resolve(undefined); };
        }),
    );
    renderDetails(makeJob({ status: 'running' }));

    const stopButton = screen.getByRole('button', { name: /Stop Job/ });
    await act(async () => {
      fireEvent.click(stopButton);
    });

    const stoppingButton = screen.getByRole('button', { name: /Stopping/ });
    expect(stoppingButton).toBeDisabled();
    fireEvent.click(stoppingButton);
    expect(mocks.cancelJob).toHaveBeenCalledTimes(1);

    resolveCancel?.();
    await waitFor(() => expect(mocks.cancelJob).toHaveBeenCalledTimes(1));
    // Let the running job's background poll settle too, to keep the test quiet.
    await waitFor(() => { expect(jobsApi.getJob).toHaveBeenCalled(); });
  });

  it('shows "No logs available" rather than a blank panel when a job has no logs yet', async () => {
    renderDetails(makeJob({ status: 'running', logs: [] }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(await screen.findByText(/No logs available/i)).toBeInTheDocument();
    await waitFor(() => { expect(jobsApi.getJob).toHaveBeenCalled(); });
  });

  it('renders log lines when present', async () => {
    renderDetails(makeJob({ status: 'failed', logs: ['INFO: starting', 'ERROR: boom'] }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(await screen.findByText(/starting/)).toBeInTheDocument();
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it('shows the job error alongside logs and result context when the job failed', () => {
    renderDetails(makeJob({ status: 'failed', error: 'Something exploded' }));
    expect(screen.getByText('Something exploded')).toBeInTheDocument();
  });

  it('preserves log wrapping and auto-scroll choices when switching tabs and copies the raw lines', async () => {
    // Tab navigation must not reset log controls or replace the copied payload with highlighted text.
    const logs = ['WARNING:trainer:loss=0.250', '[ERROR] training failed'];
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    renderDetails(makeJob({ logs }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(screen.getByText('WARNING:trainer:')).toHaveClass('text-yellow-400');
    expect(screen.getByText('0.250')).toHaveClass('text-emerald-400');
    fireEvent.click(screen.getByTitle('No wrap'));
    fireEvent.click(screen.getByTitle('Disable auto-scroll'));
    fireEvent.click(screen.getByRole('button', { name: 'Overview' }));
    fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
    expect(screen.getByTitle('Wrap lines')).toBeInTheDocument();
    expect(screen.getByTitle('Enable auto-scroll')).toBeInTheDocument();
    expect(screen.getByText('WARNING:trainer:').parentElement).toHaveClass('whitespace-pre');
    await act(async () => { fireEvent.click(screen.getByTitle('Copy all logs')); });
    expect(writeText).toHaveBeenCalledWith(logs.join('\n'));
  });

  it('prefers persisted tuning configuration over graph defaults and renders score and CV settings', () => {
    // A historical run must display the submitted settings even if its graph carries different defaults.
    renderDetails(makeJob({
      job_type: 'tuning', status: 'completed',
      graph: { nodes: [{ node_id: 'node-1', params: { tuning_config: { strategy: 'random' } } }] },
      config: { tuning_config: { strategy: 'optuna', metric: 'accuracy', n_trials: 12, cv_enabled: true, cv_type: 'stratified', cv_folds: 4, cv_shuffle: true } },
      result: { best_score: 0.875, best_params: { max_depth: 3 } },
    }));
    expect(screen.getByText('optuna')).toBeInTheDocument();
    expect(screen.queryByText('random')).not.toBeInTheDocument();
    expect(screen.getByText('sampler: tpe · pruner: median (defaults)')).toBeInTheDocument();
    expect(screen.getByText('stratified')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('0.8750')).toBeInTheDocument();
    expect(screen.getByText(/"max_depth": 3/)).toBeInTheDocument();
  });

  it('falls back to the matching graph node when the persisted tuning configuration is empty', () => {
    // Legacy jobs retain their graph settings and strategy defaults when no submitted configuration was saved.
    renderDetails(makeJob({
      job_type: 'tuning', status: 'completed', config: { tuning_config: {} }, result: {},
      graph: { nodes: [
        { node_id: 'other-node', params: { tuning_config: { strategy: 'random' } } },
        { node_id: 'node-1', params: { tuning_config: { search_strategy: 'halving_grid' } } },
      ] },
    }));
    expect(screen.getByText('halving_grid')).toBeInTheDocument();
    expect(screen.getByText('factor: 3 · resource: n_samples · min_resources: exhaust (defaults)')).toBeInTheDocument();
    expect(screen.queryByText('CV Method:')).not.toBeInTheDocument();
  });

  it('preserves voting ensemble configuration and explicit strategy parameters', () => {
    // Ensemble structure belongs to the submitted tuning config and must remain visible beside search settings.
    renderDetails(makeJob({
      job_type: 'tuning', status: 'completed', model_type: 'voting_classifier', graph: {}, result: {},
      config: { tuning_config: {
        strategy: 'optuna', strategy_params: { sampler: 'random', seed: 42 },
        base_estimators: ['random_forest', 'logistic_regression'], voting: 'soft', weights: [2, 1],
        n_jobs: -1, calibrate_base_models: true, calibration_method: 'isotonic',
      } },
    }));
    expect(screen.getByText('sampler: random · seed: 42')).toBeInTheDocument();
    expect(screen.getByText('Random Forest, Logistic Regression')).toBeInTheDocument();
    expect(screen.getByText('Random Forest: 2, Logistic Regression: 1')).toBeInTheDocument();
    expect(screen.getByText('soft')).toBeInTheDocument();
    expect(screen.getByText('All cores')).toBeInTheDocument();
    expect(screen.getByText('Isotonic')).toBeInTheDocument();
    expect(screen.queryByText('Final Estimator:')).not.toBeInTheDocument();
  });

  it('preserves stacking-specific settings without showing stale voting configuration', () => {
    // Mutually exclusive ensemble settings must not leak from graph defaults into the displayed run.
    renderDetails(makeJob({
      job_type: 'tuning', status: 'completed', model_type: 'stacking_regressor', graph: {}, result: {},
      config: { tuning_config: {
        base_estimators: ['random_forest'], final_estimator: 'ridge', passthrough: true,
        voting: 'soft', weights: [1],
      } },
    }));
    expect(screen.getByText('Ridge')).toBeInTheDocument();
    expect(screen.getByText('Passthrough:').parentElement).toHaveTextContent('Yes');
    expect(screen.queryByText('Voting:')).not.toBeInTheDocument();
    expect(screen.queryByText('Model Weights:')).not.toBeInTheDocument();
  });

  it('shows the five most important features using nested result metrics before legacy values', () => {
    // Extraction must preserve ranking, the five-feature limit, and the source precedence.
    renderDetails(makeJob({ status: 'completed', result: {
      feature_importances: { legacy: 1 },
      metrics: { feature_importances: { sixth: 0.01, fifth: 0.1, fourth: 0.2, third: 0.3, second: 0.4, first: 0.5 } },
    } }));
    const first = screen.getByTitle('first');
    const featureList = first.parentElement!.parentElement!;
    expect(within(featureList).getAllByTitle(/first|second|third|fourth|fifth/).map((element) => element.textContent))
      .toEqual(['first', 'second', 'third', 'fourth', 'fifth']);
    expect(screen.queryByTitle('sixth')).not.toBeInTheDocument();
    expect(screen.queryByTitle('legacy')).not.toBeInTheDocument();
  });

  it('lets the user hide CV metrics in the results grid', () => {
    renderDetails(makeJob({
      job_type: 'training',
      status: 'completed',
      result: { metrics: { test_f1_weighted: 0.91, cv_f1_weighted_mean: 0.9 } },
    }));
    expect(screen.getByText('cv f1 weighted mean')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox', { name: /show cv metrics/i }));
    expect(screen.queryByText('cv f1 weighted mean')).not.toBeInTheDocument();
    expect(screen.getByText('test f1 weighted')).toBeInTheDocument();
  });

  it('links the dataset using the shared RecordLink primitive', () => {
    renderDetails(
      makeJob({ dataset_id: 'ds-1', dataset_name: 'Sales Data', pipeline_id: 'pipe-42' }),
      { origin: '/jobs', filters: { tab: 'classification' } },
    );
    const datasetLink = screen.getByRole('link', { name: 'Dataset ds-1' });
    expect(datasetLink.getAttribute('href')).toContain('oc.kind=dataset');
    expect(datasetLink.getAttribute('href')).toContain('oc.origin=%2Fjobs');
  });

  it('opens a read-only node inspector for the job\'s own node instead of the canvas', async () => {
    vi.mocked(monitoringApi.getJobNode).mockResolvedValue({
      job_id: 'job-1234567890',
      node_id: 'node-1',
      node_found: true,
      node: {
        node_id: 'node-1',
        step_type: 'training',
        label: 'Training',
        params: { algorithm: 'RandomForest' },
        upstream: [],
        downstream: [],
      },
      pipeline_id: 'pipe-42',
      dataset_source_id: 'ds-1',
      run_mode: 'fixed',
      model_type: 'RandomForest',
      status: 'completed',
      is_synthetic_pipeline: false,
      can_open_in_canvas: true,
      recent_logs: [],
    });
    renderDetails(makeJob({ pipeline_id: 'pipe-42', node_id: 'node-1' }));

    fireEvent.click(screen.getByRole('button', { name: /node node-1/i }));

    expect(await screen.findByText('Training')).toBeInTheDocument();
    expect(monitoringApi.getJobNode).toHaveBeenCalledWith('job-1234567890', 'node-1');
    // Confirms this opened an in-place inspector, not a canvas page navigation.
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  it('reveals pipeline run context on activation instead of linking into a dead end', () => {
    renderDetails(makeJob({ pipeline_id: 'pipe-42' }));

    const expander = screen.getByRole('button', { name: /Pipeline run/ });
    expect(expander).toHaveAttribute('aria-expanded', 'false');
    // Not a link — there's no saved-pipeline route this id can resolve to.
    expect(screen.queryByRole('link', { name: /pipe-42/ })).not.toBeInTheDocument();

    fireEvent.click(expander);

    expect(expander).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('pipe-42')).toBeInTheDocument();
    expect(screen.getByText(/not a separately saved pipeline/)).toBeInTheDocument();
  });

  it('explains a preview/branch pipeline id as a non-navigable synthetic run', () => {
    const previewId = 'preview_275b9aa6-7636-4642-afc1-226ea420ddc6__branch_0';
    renderDetails(makeJob({ pipeline_id: previewId }));

    const expander = screen.getByRole('button', { name: /Preview run/ });
    fireEvent.click(expander);

    expect(screen.getByText(previewId)).toBeInTheDocument();
    expect(screen.getByText(/Preview run — not a saved pipeline/)).toBeInTheDocument();
    expect(screen.getByText(/branch 0/)).toBeInTheDocument();
  });

  it('is keyboard operable and keeps the full pipeline id available via a copy control', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    renderDetails(makeJob({ pipeline_id: 'pipe-99' }));

    const expander = screen.getByRole('button', { name: /Pipeline run/ });
    expander.focus();
    expect(expander).toHaveFocus();
    fireEvent.click(expander);

    const copyButton = screen.getByRole('button', { name: 'Copy run id' });
    fireEvent.click(copyButton);
    expect(writeText).toHaveBeenCalledWith('pipe-99');
    expect(await screen.findByRole('button', { name: 'Run id copied' })).toBeInTheDocument();
  });

  it('shows a passed leakage-gate verdict tile for jobs that ran through the gate', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: { metrics: { leakage_gate: { status: 'passed', messages: [] } } },
    }));
    const label = screen.getByText('Leakage Gate');
    expect(within(label.parentElement as HTMLElement).getByText('Passed')).toBeInTheDocument();
  });

  it('flags jobs trained without a train/test split in the gate tile', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          leakage_gate: {
            status: 'no_split',
            messages: ['No train/test split is defined in this pipeline graph.'],
          },
        },
      },
    }));
    const label = screen.getByText('Leakage Gate');
    expect(within(label.parentElement as HTMLElement).getByText('No split')).toBeInTheDocument();
  });

  it('omits the gate tile for legacy jobs without a verdict', () => {
    renderDetails(makeJob({ status: 'completed', result: { metrics: { accuracy: 0.9 } } }));
    expect(screen.queryByText('Leakage Gate')).not.toBeInTheDocument();
  });

  it('shows a verified fold-refit-audit tile with the row ratio when isolation held', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          fold_refit_audit: {
            fit_calls: 11,
            max_fit_rows: 320,
            transform_calls: 10,
            train_rows: 320,
            isolation_ok: true,
          },
        },
      },
    }));
    const label = screen.getByText('Fold Refit Audit');
    expect(
      within(label.parentElement as HTMLElement).getByText('Isolation verified (320/320)')
    ).toBeInTheDocument();
  });

  it('flags the audit tile when a fit saw more rows than the train split', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          fold_refit_audit: {
            fit_calls: 5,
            max_fit_rows: 400,
            transform_calls: 5,
            train_rows: 320,
            isolation_ok: false,
          },
        },
      },
    }));
    const label = screen.getByText('Fold Refit Audit');
    expect(
      within(label.parentElement as HTMLElement).getByText('Isolation warning')
    ).toBeInTheDocument();
  });

  it('omits the audit tile when no fold-refit audit was stamped', () => {
    renderDetails(makeJob({ status: 'completed', result: { metrics: { accuracy: 0.9 } } }));
    expect(screen.queryByText('Fold Refit Audit')).not.toBeInTheDocument();
  });

  it('opens the audit modal with the measured fold statistics on tile click', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          fold_refit_audit: {
            fit_calls: 11,
            max_fit_rows: 320,
            transform_calls: 10,
            train_rows: 320,
            isolation_ok: true,
          },
        },
      },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Fold Refit Audit/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('Measured during this run')).toBeInTheDocument();
    expect(within(dialog).getByText('11 call(s)')).toBeInTheDocument();
    expect(within(dialog).getByText('320 of 320 train rows')).toBeInTheDocument();
    expect(within(dialog).getByText('10 call(s)')).toBeInTheDocument();
    expect(within(dialog).getByText(/no held-out rows in any fit/i)).toBeInTheDocument();
  });

  it('shows the held-out-rows warning wording in the audit modal when isolation failed', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          fold_refit_audit: {
            fit_calls: 5,
            max_fit_rows: 400,
            transform_calls: 5,
            train_rows: 320,
            isolation_ok: false,
          },
        },
      },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Fold Refit Audit/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText(/held-out \(validation\/test\) rows may have entered a fit/)).toBeInTheDocument();
    expect(within(dialog).getByText(/held-out rows may have entered a fit/i)).toBeInTheDocument();
  });

  it('shows a score-advisory tile for jobs that fell back to pre-transformed scoring', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: { metrics: { fold_refit_fallback: 'row_changing_branch_step' } },
    }));
    const label = screen.getByText('Score Advisory');
    expect(
      within(label.parentElement as HTMLElement).getByText('Scores may be optimistic')
    ).toBeInTheDocument();
  });

  it('opens the advisory modal explaining the fallback reason on tile click', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: { metrics: { fold_refit_fallback: 'row_changing_branch_step' } },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Score Advisory/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('row_changing_branch_step')).toBeInTheDocument();
    expect(within(dialog).getByText(/branches no longer align row-for-row/)).toBeInTheDocument();
    expect(within(dialog).getByText(/optimistically biased/)).toBeInTheDocument();
  });

  it('omits the score-advisory tile when the run did not fall back', () => {
    renderDetails(makeJob({ status: 'completed', result: { metrics: { accuracy: 0.9 } } }));
    expect(screen.queryByText('Score Advisory')).not.toBeInTheDocument();
  });

  it('opens a verdict modal listing what the gate checked and why exemptions were allowed', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          leakage_gate: {
            status: 'passed',
            messages: [],
            splitters: ['split'],
            checked: [
              { node_id: 'scale', step_type: 'StandardScaler', before_split: false, violation: false },
            ],
            exempted: [
              { node_id: 'fill', step_type: 'SimpleImputer', reason: 'Constant imputation — nothing learned from rows.' },
            ],
          },
        },
      },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Leakage Gate/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('StandardScaler')).toBeInTheDocument();
    expect(within(dialog).getByText(/runs after the split/i)).toBeInTheDocument();
    expect(within(dialog).getByText('SimpleImputer')).toBeInTheDocument();
    expect(within(dialog).getByText(/Constant imputation/)).toBeInTheDocument();
  });

  it('flags violating nodes in the modal for a warnings verdict', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          leakage_gate: {
            status: 'warnings',
            messages: ['Data leakage risk: node \'scale\' (StandardScaler) fits on the whole dataset.'],
            splitters: ['split'],
            checked: [
              { node_id: 'scale', step_type: 'StandardScaler', before_split: true, violation: true },
            ],
            exempted: [],
          },
        },
      },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Leakage Gate/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText('StandardScaler')).toBeInTheDocument();
    expect(within(dialog).getByText(/fits before the split/i)).toBeInTheDocument();
    expect(within(dialog).getByText(/Data leakage risk/)).toBeInTheDocument();
  });

  it('falls back to the recorded messages for legacy verdicts without detail', () => {
    renderDetails(makeJob({
      status: 'completed',
      result: {
        metrics: {
          leakage_gate: {
            status: 'no_split',
            messages: ['No train/test split is defined in this pipeline graph.'],
          },
        },
      },
    }));

    fireEvent.click(screen.getByRole('button', { name: /Leakage Gate/ }));

    const dialog = screen.getByRole('dialog');
    expect(within(dialog).getByText(/No train\/test split is defined/)).toBeInTheDocument();
  });

  describe('tuning trial chart', () => {
    const trials = [
      { params: { C: 0.1 }, score: 0.6 },
      { params: { C: 1 }, score: 0.8 },
      { params: { C: 10 }, score: 0.75 },
    ];

    it('redraws the trial chart for a completed tuning job from persisted metrics', () => {
      renderDetails(makeJob({
        job_type: 'tuning',
        status: 'completed',
        metrics: { trials } as unknown as Record<string, number>,
        result: { scoring_metric: 'accuracy' },
      }));

      expect(screen.getByText('Tuning Trials')).toBeInTheDocument();
      expect(screen.getByText(/accuracy/)).toBeInTheDocument();
      expect(screen.getByText('Best so far')).toBeInTheDocument();
    });

    it('shows no chart for a fixed run with a single trial', () => {
      renderDetails(makeJob({
        job_type: 'training',
        status: 'completed',
        metrics: { trials: trials.slice(0, 1) } as unknown as Record<string, number>,
      }));

      expect(screen.queryByText('Tuning Trials')).toBeNull();
    });

    it('shows no chart for a running job without trial data yet', () => {
      renderDetails(makeJob({ job_type: 'tuning', status: 'running' }));

      expect(screen.queryByText('Tuning Trials')).toBeNull();
    });

    it('redraws the iteration chart for a completed fixed boosting job', () => {
      renderDetails(makeJob({
        job_type: 'training',
        status: 'completed',
        metrics: {
          trials: [{ params: {}, score: 0.8 }],
          iterations: [
            { iteration: 1, total: 200, score: 0.5, metric: 'logloss', direction: 'minimize' },
            { iteration: 2, total: 200, score: 0.4, metric: 'logloss', direction: 'minimize' },
          ],
          iteration_direction: 'minimize',
        } as unknown as Record<string, number>,
      }));

      expect(screen.getByText('Boosting Iterations')).toBeInTheDocument();
      expect(screen.queryByText('Tuning Trials')).toBeNull();
    });

    it('shows no iteration chart for a single-iteration boosting job', () => {
      renderDetails(makeJob({
        job_type: 'training',
        status: 'completed',
        metrics: {
          trials: [{ params: {}, score: 0.8 }],
          iterations: [
            { iteration: 1, total: 1, score: 0.5, metric: 'logloss', direction: 'minimize' },
          ],
        } as unknown as Record<string, number>,
      }));

      expect(screen.queryByText('Boosting Iterations')).toBeNull();
    });

    it('offers Trials/Iterations tabs when a boosting tuning job persisted both series', () => {
      renderDetails(makeJob({
        job_type: 'tuning',
        status: 'completed',
        metrics: {
          trials,
          iterations: [
            { iteration: 1, total: 3, score: 0.6, metric: 'logloss', direction: 'minimize' },
            { iteration: 2, total: 3, score: 0.4, metric: 'logloss', direction: 'minimize' },
            { iteration: 3, total: 3, score: 0.5, metric: 'logloss', direction: 'minimize' },
          ],
          iteration_direction: 'minimize',
        } as unknown as Record<string, number>,
      }));

      // Default follows the active series: iterations for a boosting job.
      expect(screen.getByRole('button', { name: 'Trials' })).toBeInTheDocument();
      const iterationsTab = screen.getByRole('button', { name: 'Iterations' });
      expect(iterationsTab).toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByText('Boosting Iterations')).toBeInTheDocument();
      expect(screen.queryByText('Tuning Trials')).toBeNull();

      // Clicking Trials pins the trial chart.
      fireEvent.click(screen.getByRole('button', { name: 'Trials' }));
      expect(screen.getByText('Tuning Trials')).toBeInTheDocument();
      expect(screen.queryByText('Boosting Iterations')).toBeNull();
      expect(screen.getByRole('button', { name: 'Trials' })).toHaveAttribute('aria-pressed', 'true');
    });

    it('renders no series tabs for single-series jobs', () => {
      renderDetails(makeJob({
        job_type: 'tuning',
        status: 'completed',
        metrics: { trials } as unknown as Record<string, number>,
      }));

      expect(screen.getByText('Tuning Trials')).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Trials' })).toBeNull();
      expect(screen.queryByRole('button', { name: 'Iterations' })).toBeNull();
    });

    it('preserves a pinned chart across tab switches and resets it for a different job', () => {
      // Keeping chart state mounted preserves user choice, while a new run resumes automatic series selection.
      const job = makeJob({
        job_type: 'tuning', status: 'completed',
        metrics: {
          trials,
          iterations: [
            { iteration: 1, score: 0.6, metric: 'logloss', direction: 'minimize' },
            { iteration: 2, score: 0.4, metric: 'logloss', direction: 'minimize' },
          ],
        } as unknown as Record<string, number>,
      });
      const onBack = vi.fn();
      const onClose = vi.fn();
      const { rerender } = render(<MemoryRouter><JobDetailsView job={job} onBack={onBack} onClose={onClose} /></MemoryRouter>);
      fireEvent.click(screen.getByRole('button', { name: 'Trials' }));
      fireEvent.click(screen.getByRole('button', { name: /Live Logs/ }));
      fireEvent.click(screen.getByRole('button', { name: 'Overview' }));
      expect(screen.getByRole('button', { name: 'Trials' })).toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByText('Tuning Trials')).toBeInTheDocument();

      rerender(<MemoryRouter><JobDetailsView job={{ ...job, job_id: 'job-new-series' }} onBack={onBack} onClose={onClose} /></MemoryRouter>);
      expect(screen.getByRole('button', { name: 'Iterations' })).toHaveAttribute('aria-pressed', 'true');
      expect(screen.getByText('Boosting Iterations')).toBeInTheDocument();
    });
  });
});
