import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { jobsApi } from '../../../core/api/jobs';
import { registryApi } from '../../../core/api/registry';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { useJobStore } from '../../../core/store/useJobStore';
import { TrainingSettings, type TrainingConfig, type TrainingTask } from './TrainingSettings';

vi.mock('../../../core/hooks/useTrainingNodeContext', () => ({ useTrainingNodeContext: vi.fn() }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));
vi.mock('../../../core/api/registry', () => ({ registryApi: { getAllNodes: vi.fn() } }));
vi.mock('../../../core/api/jobs', () => ({ jobsApi: {
  getHyperparameters: vi.fn(), getDefaultSearchSpace: vi.fn(), getTuningHistory: vi.fn(),
} }));

const config: TrainingConfig = {
  run_mode: 'basic', target_column: 'outcome', model_type: 'random_forest_classifier',
  hyperparameters: {}, search_space: {}, n_trials: 10, metric: 'accuracy', search_strategy: 'random',
  random_state: 42, cv_enabled: false, cv_folds: 5, cv_type: 'k_fold', cv_shuffle: true, cv_random_state: 42,
};

function trainingContext(patch: Partial<ReturnType<typeof useTrainingNodeContext>> = {}) {
  return {
    availableColumns: [], upstreamTarget: undefined, datasetId: 'dataset-1',
    runJob: vi.fn().mockResolvedValue(undefined), isSubmitting: false,
    submissionMessage: '', runFeedback: null, ...patch,
  };
}

async function renderSettings(patch: Partial<TrainingConfig> = {}, task?: TrainingTask) {
  await act(async () => {
    render(<TrainingSettings config={{ ...config, ...patch }} onChange={vi.fn()} nodeId="model" {...(task ? { task } : {})} />);
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('IntersectionObserver', undefined);
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext());
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([
    { id: 'random_forest_classifier', name: 'Random Forest Classifier', category: 'Modeling', description: '', params: {}, tags: ['classification'] },
    { id: 'ridge_regression', name: 'Ridge Regression', category: 'Modeling', description: '', params: {}, tags: ['regression'] },
  ]);
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([]);
  vi.mocked(jobsApi.getDefaultSearchSpace).mockResolvedValue({});
  vi.mocked(jobsApi.getTuningHistory).mockResolvedValue([]);
  useJobStore.setState({ jobs: [], runJobs: {}, inspectedRun: null, toggleDrawer: vi.fn(), setTab: vi.fn() });
});

afterEach(() => vi.unstubAllGlobals());

/** The footer must explain the first unmet prerequisite without submitting invalid work. */
it.each([
  { name: 'missing dataset', datasetId: undefined, patch: {}, reason: /Connect a dataset node upstream/ },
  { name: 'missing target', datasetId: 'dataset-1', patch: { target_column: '' }, reason: /Choose a target column/ },
  { name: 'whitespace target', datasetId: 'dataset-1', patch: { target_column: '  ' }, reason: /Choose a target column/ },
  { name: 'missing model', datasetId: 'dataset-1', patch: { model_type: '' }, reason: /Choose a model/ },
])('blocks training with $name', async ({ datasetId, patch, reason }) => {
  const context = trainingContext({ datasetId });
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings(patch);
  const action = screen.getByRole('button', { name: 'Train model' });
  expect(action).toBeDisabled();
  expect(action).toHaveAccessibleDescription(reason);
  fireEvent.click(action);
  expect(context.runJob).not.toHaveBeenCalled();
});

/** The selected mode and task must determine submission and which model history opens. */
it.each([
  { run_mode: 'basic' as const, model_type: 'random_forest_classifier', task: 'classification' as const, jobType: 'training', name: 'Train model' },
  { run_mode: 'advanced' as const, model_type: 'ridge_regression', task: 'regression' as const, jobType: 'tuning', name: 'Tune model' },
  { run_mode: 'advanced' as const, model_type: 'random_forest_classifier', task: 'text_classification' as const, jobType: 'tuning', name: 'Tune model' },
])('submits $run_mode work to $task history', async ({ run_mode, model_type, task, jobType, name }) => {
  const context = trainingContext();
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings({ run_mode, model_type }, task);
  const action = screen.getByRole('button', { name });
  expect(action).toBeEnabled();
  expect(action).toHaveAccessibleDescription(/in the background/);
  await act(async () => fireEvent.click(action));
  expect(context.runJob).toHaveBeenCalledExactlyOnceWith(jobType, task);
});

/** Generic nodes must infer regression history and safely route unknown models. */
it.each([
  { model_type: 'ridge_regression', expectedTask: 'regression' },
  { model_type: 'custom_model', expectedTask: 'classification' },
])('resolves history for the generic $model_type model', async ({ model_type, expectedTask }) => {
  const context = trainingContext();
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings({ model_type });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Train model' })));
  expect(context.runJob).toHaveBeenCalledExactlyOnceWith('training', expectedTask);
});

/** In-flight work must stay visible and cannot be accidentally submitted a second time. */
it('shows pending feedback while disabling the training action', async () => {
  const context = trainingContext({ isSubmitting: true, submissionMessage: 'Random forest: Submitting...' });
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings();
  const action = screen.getByRole('button', { name: 'Submitting job...' });
  expect(action).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Random forest: Submitting...');
  fireEvent.click(action);
  expect(context.runJob).not.toHaveBeenCalled();
});

/** Successful node actions must retain their receipt and reopen task-scoped history. */
it('opens regression history from a submitted run', async () => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({
    runFeedback: { label: 'Ridge training', jobIds: ['ridge-job'] },
  }));
  useJobStore.setState({ inspectedRun: { label: 'Previous pipeline', jobIds: ['old-job'] } });
  await renderSettings({ model_type: 'ridge_regression' });
  expect(screen.getByRole('status')).toHaveTextContent('Ridge training: Awaiting status');
  fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
  expect(useJobStore.getState().inspectedRun).toBeNull();
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('regression');
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
});

/** Advanced tuning history must open read-only so prior params cannot overwrite a search. */
it('opens best-parameter history in advanced mode', async () => {
  await renderSettings({ run_mode: 'advanced' });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'View Best Parameters History' })));
  expect(screen.getByRole('heading', { name: 'Best Parameters History' })).toBeVisible();
  expect(screen.getByText('View parameters for:')).toBeVisible();
  expect(jobsApi.getTuningHistory).toHaveBeenCalledWith('random_forest_classifier');
  expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
});
