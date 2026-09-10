import { act, fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
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

/** Controlled updates exercise the same effect and focus lifetimes as the node inspector. */
function SettingsHarness({ initial }: { initial: TrainingConfig }) {
  const [value, setValue] = useState(initial);
  return <TrainingSettings config={value} onChange={setValue} nodeId="model" />;
}

/** Basic training keeps a manual target while advanced tuning follows its upstream target. */
it('preserves a manual basic target and synchronizes it on switching to tuning', async () => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ upstreamTarget: 'upstream' }));
  await act(async () => render(<SettingsHarness initial={config} />));
  expect(screen.getByRole('textbox', { name: 'Target Column' })).toHaveValue('outcome');
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Advanced (Tuning)' })));
  expect(screen.getByRole('textbox', { name: 'Target Column' })).toHaveValue('upstream');
});

/** Editing a parameter must retain focus, and a model switch must load that model's defaults. */
it('keeps customized inputs mounted and reinitializes customization on a model switch', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockImplementation(async model => model === 'ridge_regression'
    ? [{ name: 'alpha', label: 'Alpha', type: 'number', default: 1 }]
    : [{ name: 'depth', label: 'Depth', type: 'number', default: 3 }]);
  await act(async () => render(<SettingsHarness initial={{ ...config, hyperparameters: { depth: 3 } }} />));
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  const depth = screen.getByRole('textbox', { name: 'Depth' });
  depth.focus();
  fireEvent.change(depth, { target: { value: '7' } });
  fireEvent.blur(depth);
  depth.focus();
  expect(screen.getByRole('textbox', { name: 'Depth' })).toBe(depth);
  expect(depth).toHaveFocus();
  fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
  await act(async () => fireEvent.change(screen.getByRole('combobox', { name: 'Model Type' }), { target: { value: 'ridge_regression' } }));
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByRole('checkbox', { name: 'Customize' })).toBeChecked();
  expect(screen.getByRole('textbox', { name: 'Alpha' })).toHaveValue('1');
});

/** Only crossing the grid strategy boundary replaces search values, and strategy settings never leak across selections. */
it('reloads search defaults only when changing strategy class and clears customized strategy params', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([{ name: 'depth', label: 'Depth', type: 'number', default: 3 }]);
  await act(async () => render(<SettingsHarness initial={{ ...config, run_mode: 'advanced', search_strategy: 'optuna', strategy_params: { sampler: 'random' } }} />));
  expect(screen.getByText(/sampler: random/)).toBeVisible();
  await act(async () => fireEvent.change(screen.getByRole('combobox', { name: 'Search Method' }), { target: { value: 'halving_random' } }));
  expect(jobsApi.getDefaultSearchSpace).toHaveBeenCalledTimes(1);
  expect(screen.getByText(/Using defaults/)).toBeVisible();
  await act(async () => fireEvent.change(screen.getByRole('combobox', { name: 'Search Method' }), { target: { value: 'grid' } }));
  expect(jobsApi.getDefaultSearchSpace).toHaveBeenLastCalledWith('random_forest_classifier', 'grid');
  expect(jobsApi.getDefaultSearchSpace).toHaveBeenCalledTimes(2);
  expect(screen.getByRole('spinbutton', { name: 'Trials' })).toBeDisabled();
});

/** CV expansion survives tab switches, temporal columns sort first, and a zero seed is preserved. */
it('preserves CV controls across tabs and time-series selection', async () => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ availableColumns: [
    { name: 'amount', dtype: 'float64', missing_count: 0, missing_ratio: 0, unique_count: 2 },
    { name: 'created', dtype: 'datetime64', missing_count: 0, missing_ratio: 0, unique_count: 2 },
  ] }));
  await act(async () => render(<SettingsHarness initial={{ ...config, cv_enabled: true, cv_random_state: 0 }} />));
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  expect(screen.getByRole('spinbutton', { name: 'Fold Split Seed' })).toHaveValue(0);
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  fireEvent.click(screen.getByRole('button', { name: 'Configuration' }));
  expect(screen.getByRole('button', { name: 'Cross Validation' })).toHaveAttribute('aria-expanded', 'true');
  fireEvent.change(screen.getByRole('combobox', { name: 'Method' }), { target: { value: 'time_series_split' } });
  expect(screen.queryByRole('spinbutton', { name: 'Fold Split Seed' })).not.toBeInTheDocument();
  const timeColumn = screen.getByRole('combobox', { name: 'Time Column (optional)' });
  expect(Array.from(timeColumn.querySelectorAll('option'), option => option.value)).toEqual(['', 'created', 'amount']);
});

/** Threshold tuning belongs only to classifiers, with the supplied seed and checkbox value retained. */
it.each([
  { model_type: 'random_forest_classifier', visible: true },
  { model_type: 'ridge_regression', visible: false },
])('shows threshold tuning appropriately for $model_type', async ({ model_type, visible }) => {
  await renderSettings({ run_mode: 'advanced', model_type, random_state: 0, tune_threshold: true });
  expect(screen.getByRole('spinbutton', { name: 'Random State' })).toHaveValue(0);
  const threshold = screen.queryByRole('checkbox', { name: 'Tune decision threshold' });
  if (visible) expect(threshold).toBeChecked();
  else expect(threshold).not.toBeInTheDocument();
});

/** Conditional hyperparameters need names even after changing settings sections. */
it('labels basic hyperparameters and exposes the active training mode', async () => {
  vi.mocked(jobsApi.getHyperparameters).mockResolvedValue([
    { name: 'penalty', label: 'Penalty', type: 'select', default: 'elasticnet', options: [{ label: 'Elastic Net', value: 'elasticnet' }] },
    { name: 'l1_ratio', label: 'L1 Ratio', type: 'number', default: 0.5, depends_on: { param: 'penalty', value: 'elasticnet' } },
  ]);
  await renderSettings({ hyperparameters: { penalty: 'elasticnet', l1_ratio: 0.5 } });
  expect(screen.getByRole('button', { name: 'Basic', pressed: true })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Hyperparameters' }));
  expect(screen.getByRole('combobox', { name: 'Penalty' })).toBeVisible();
  expect(screen.getByRole('textbox', { name: 'L1 Ratio' })).toHaveValue('0.5');
});

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
