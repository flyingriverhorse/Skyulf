import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import type { ColumnProfile } from '../../../core/api/client';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { registryApi } from '../../../core/api/registry';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
import { convertEnsembleNode } from '../../../core/utils/pipelineConversion/ensemble';
import { EnsembleSettings, type EnsembleConfig } from './EnsembleSettings';

vi.mock('../../../core/hooks/useTrainingNodeContext', () => ({ useTrainingNodeContext: vi.fn() }));
vi.mock('../../../core/hooks/useIsWideContainer', () => ({ useIsWideContainer: () => [null, false] }));
vi.mock('../../../core/api/registry', () => ({ registryApi: { getAllNodes: vi.fn() } }));

const config: EnsembleConfig = {
  task: 'classification', strategy: 'voting', model_type: 'voting_classifier',
  base_estimators: ['random_forest', 'logistic_regression'], voting: 'soft',
  final_estimator: 'logistic_regression', cv: 3, target_column: 'outcome',
  cv_enabled: false, cv_folds: 5, cv_type: 'k_fold', cv_shuffle: true, cv_random_state: 42,
  run_mode: 'basic', search_strategy: 'random', n_trials: 10, metric: 'accuracy',
  tune_base_models: true, random_state: 42,
};

function trainingContext(patch: Partial<ReturnType<typeof useTrainingNodeContext>> = {}) {
  return {
    availableColumns: [], upstreamTarget: undefined, datasetId: 'dataset-1',
    runJob: vi.fn().mockResolvedValue(undefined), isSubmitting: false,
    submissionMessage: '', runFeedback: null, ...patch,
  };
}

async function renderSettings(patch: Partial<EnsembleConfig> = {}) {
  const onChange = vi.fn<(next: EnsembleConfig) => void>();
  await act(async () => {
    render(<EnsembleSettings config={{ ...config, ...patch }} onChange={onChange} nodeId="ensemble" />);
  });
  return onChange;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal('IntersectionObserver', undefined);
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([]);
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext());
  useGraphStore.setState({ nodes: [], edges: [] });
  useJobStore.setState({ jobs: [], runJobs: {}, inspectedRun: null, toggleDrawer: vi.fn(), setTab: vi.fn() });
});

afterEach(() => vi.unstubAllGlobals());

/** Ensemble controls need names and selection state across basic and advanced modes. */
it('labels voting weights, target and conditional tuning fields', async () => {
  await renderSettings({ run_mode: 'advanced', calibrate_base_models: true, cv_enabled: true });
  expect(screen.getByRole('button', { name: 'Voting', pressed: true })).toBeVisible();
  expect(screen.getByRole('group', { name: 'Base Models' })).toBeVisible();
  for (const name of ['Random Forest weight', 'Logistic Regression weight', 'Trials', 'Random State', 'Calibration CV Folds']) {
    expect(screen.getByRole('spinbutton', { name })).toBeVisible();
  }
  for (const name of ['Parallel Jobs', 'Calibration Method', 'Search Strategy', 'Optimize Metric']) {
    expect(screen.getByRole('combobox', { name })).toBeVisible();
  }
  expect(screen.getByRole('textbox', { name: 'Target Column' })).toBeVisible();
  expect(screen.getByRole('checkbox', { name: 'Tune base model hyperparameters' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  expect(screen.getByRole('spinbutton', { name: 'Folds' })).toBeVisible();
  expect(screen.getByRole('combobox', { name: 'Method' })).toBeVisible();
  expect(screen.getByRole('spinbutton', { name: 'Fold Split Seed' })).toBeVisible();
});

/** Stacking and time-series options must remain named when their sections appear. */
it('labels stacking and time-series controls', async () => {
  await renderSettings({ strategy: 'stacking', cv_enabled: true, cv_type: 'time_series_split' });
  expect(screen.getByRole('combobox', { name: 'Final Estimator (meta-learner)' })).toBeVisible();
  expect(screen.getByRole('spinbutton', { name: 'Stacking CV Folds' })).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  expect(screen.getByRole('combobox', { name: 'Time Column (optional)' })).toBeVisible();
});

/** Missing prerequisites must explain the correction and prevent a submission. */
it.each([
  { name: 'disconnected dataset', patch: {}, datasetId: undefined, reason: /Connect a dataset node upstream/ },
  { name: 'empty target', patch: { target_column: '' }, datasetId: 'dataset-1', reason: /Choose a target column/ },
  { name: 'whitespace target', patch: { target_column: '  ' }, datasetId: 'dataset-1', reason: /Choose a target column/ },
  { name: 'no base models', patch: { base_estimators: [] }, datasetId: 'dataset-1', reason: /Choose at least two base models/ },
  { name: 'one base model', patch: { base_estimators: ['random_forest'] }, datasetId: 'dataset-1', reason: /Choose at least two base models/ },
])('blocks an ensemble with $name', async ({ patch, datasetId, reason }) => {
  const context = trainingContext({ datasetId });
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings(patch);
  const action = screen.getByRole('button', { name: 'Train ensemble' });
  expect(action).toBeDisabled();
  expect(action).toHaveAccessibleDescription(reason);
  fireEvent.click(action);
  expect(context.runJob).not.toHaveBeenCalled();
});

/** Basic and advanced ensemble actions must reach the correct job type and history tab. */
it.each([
  { run_mode: 'basic' as const, name: 'Train ensemble', jobType: 'training', description: /Trains the voting ensemble with 2 base models/ },
  { run_mode: 'advanced' as const, name: 'Tune ensemble', jobType: 'tuning', description: /Tunes the voting ensemble with 2 base models/ },
])('submits $run_mode ensembles', async ({ run_mode, name, jobType, description }) => {
  const context = trainingContext();
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings({ run_mode });
  const action = screen.getByRole('button', { name });
  expect(action).toBeEnabled();
  expect(action).toHaveAccessibleDescription(description);
  fireEvent.click(action);
  expect(context.runJob).toHaveBeenCalledExactlyOnceWith(jobType, 'ensemble');
});

/** Pending feedback must replace the action label and block repeated submissions. */
it('announces an in-flight submission and disables the action', async () => {
  const context = trainingContext({ isSubmitting: true, submissionMessage: 'Voting ensemble: Submitting...' });
  vi.mocked(useTrainingNodeContext).mockReturnValue(context);
  await renderSettings();
  const action = screen.getByRole('button', { name: 'Submitting job...' });
  expect(action).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Voting ensemble: Submitting...');
  fireEvent.click(action);
  expect(context.runJob).not.toHaveBeenCalled();
});

/** A completed POST must keep its receipt accessible and reopen ensemble history. */
it('opens the ensemble history from its submitted run', async () => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({
    runFeedback: { label: 'Voting ensemble', jobIds: ['ensemble-job'] },
  }));
  useJobStore.setState({ inspectedRun: { label: 'Previous pipeline', jobIds: ['old-job'] } });
  await renderSettings();
  expect(screen.getByRole('status')).toHaveTextContent('Voting ensemble: Awaiting status');
  fireEvent.click(screen.getByRole('button', { name: 'View jobs' }));
  expect(useJobStore.getState().inspectedRun).toBeNull();
  expect(useJobStore.getState().setTab).toHaveBeenCalledWith('ensemble');
  expect(useJobStore.getState().toggleDrawer).toHaveBeenCalledWith(true);
});

/** Users must be able to correct a missing target or insufficient model selection. */
it('emits target and base-model edits from the real controls', async () => {
  const onChange = await renderSettings({ target_column: '', base_estimators: ['random_forest'] });
  fireEvent.change(screen.getByPlaceholderText('Type the target column name'), { target: { value: 'outcome' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ target_column: 'outcome' }));
  fireEvent.click(screen.getByRole('button', { name: 'Logistic Regression' }));
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ base_estimators: ['random_forest', 'logistic_regression'] }));
});

function column(dtype: string, unique_count = 3): ColumnProfile {
  return { name: 'outcome', dtype, unique_count, missing_count: 0, missing_ratio: 0 };
}

function connectModels(data: Record<string, unknown>[]) {
  useGraphStore.setState({
    nodes: data.map((model, i) => ({ id: `model-${i}`, position: { x: 0, y: 0 }, data: { definitionType: 'training', ...model } })),
    edges: data.map((_, i) => ({ id: `edge-${i}`, source: `model-${i}`, target: 'ensemble' })),
  });
}

/** Task inference must retain dtype precedence and the discrete-integer cutoff. */
it.each([
  ['float64', 3, 'regression'], ['int64', 20, 'classification'],
  ['int64', 21, 'regression'], ['int64', 0, 'regression'],
  ['boolean', 2, 'classification'], ['string', 30, 'classification'],
  ['unknown', 3, null],
] as const)('infers %s with %i values as %s', async (dtype, count, task) => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ availableColumns: [column(dtype, count)] }));
  const onChange = await renderSettings();
  if (task === 'regression') {
    expect(onChange).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({
      task, model_type: 'voting_regressor', base_estimators: ['random_forest', 'gradient_boosting', 'ridge'],
      final_estimator: 'ridge', metric: 'r2',
    }));
  } else {
    expect(onChange).not.toHaveBeenCalled();
  }
});

/** Manual task changes reset task defaults while preserving unrelated configuration. */
it('emits task and strategy changes and releases the manual task lock', async () => {
  const onChange = await renderSettings({ task_manual: true, weights: { random_forest: 0 } });
  fireEvent.click(screen.getByRole('button', { name: 'Regression' }));
  expect(onChange).toHaveBeenLastCalledWith({ ...config, task_manual: true, weights: { random_forest: 0 },
    task: 'regression', model_type: 'voting_regressor', base_estimators: ['random_forest', 'gradient_boosting', 'ridge'],
    final_estimator: 'ridge', metric: 'r2' });
  fireEvent.click(screen.getByRole('button', { name: 'Stacking' }));
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ strategy: 'stacking', model_type: 'stacking_classifier' }));
  fireEvent.click(screen.getByRole('button', { name: 'Auto-detect' }));
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ task_manual: false }));
});

/** First connected task/target and first advanced model must keep their distinct precedence. */
it('synchronizes connected models with tuning, CV and merged duplicate-model parameters', async () => {
  connectModels([
    { model_type: 'ridge_regression', target_column: 'wired', cv_enabled: false, cv_folds: 7,
      cv_type: 'shuffle_split', cv_shuffle: false, cv_random_state: 0, cv_time_column: 'when', hyperparameters: { alpha: 2 } },
    { model_type: 'random_forest_regressor', run_mode: 'advanced', search_strategy: 'optuna', n_trials: 22, metric: 'mae', search_space: { max_depth: 4 } },
    { model_type: 'ridge_regression', params: { alpha: 3 } },
    { model_type: 'svc', run_mode: 'advanced', search_strategy: 'grid', n_trials: 99 },
  ]);
  const onChange = await renderSettings({ base_estimator_params: { ridge: { fit_intercept: false }, svc: { C: 7 } } });
  expect(onChange).toHaveBeenCalledExactlyOnceWith({ ...config,
    task: 'regression', model_type: 'voting_regressor', base_estimators: ['ridge', 'random_forest'],
    run_mode: 'advanced', search_strategy: 'optuna', n_trials: 22, metric: 'mae', target_column: 'wired',
    cv_enabled: false, cv_folds: 7, cv_type: 'shuffle_split', cv_shuffle: false, cv_random_state: 0, cv_time_column: 'when',
    base_estimator_params: { ridge: { fit_intercept: false, alpha: 3 }, svc: { C: 7 }, random_forest: { max_depth: 4 } },
  });
});

/** Connected CV seeds must reach fixed and tuned requests, including zero and absent-seed fallbacks. */
it.each((['basic', 'advanced'] as const).flatMap(run_mode => [
  { run_mode, seed: 0, localSeed: 42, expected: 0 },
  { run_mode, seed: 7, localSeed: 42, expected: 7 },
  { run_mode, seed: undefined, localSeed: 42, expected: 42 },
  { run_mode, seed: undefined, localSeed: 0, expected: 0 },
]))('converts connected CV seed $seed with local seed $localSeed in $run_mode mode', async ({ run_mode, seed, localSeed, expected }) => {
  connectModels([
    { model_type: 'logistic_regression', target_column: 'wired', run_mode, cv_enabled: true,
      ...(seed === undefined ? {} : { cv_random_state: seed }) },
    { model_type: 'random_forest_classifier' },
  ]);
  const onChange = await renderSettings({ run_mode, cv_enabled: true, cv_random_state: localSeed });
  expect(onChange).toHaveBeenCalledTimes(1);
  const synchronized = onChange.mock.calls[0]?.[0];
  if (!synchronized) throw new Error('Connected model settings were not synchronized');
  const { nodes, edges } = useGraphStore.getState();
  const ensemble = { id: 'ensemble', position: { x: 0, y: 0 }, data: { ...synchronized } };
  const converted = convertEnsembleNode(ensemble, nodes, edges, edges, nodes.map(node => node.id));
  const expectedParams = run_mode === 'advanced'
    ? { run_mode: 'tuned', tuning_config: expect.objectContaining({ cv_random_state: expected }) }
    : { run_mode: 'fixed', cv_random_state: expected };
  expect(converted.params).toEqual(expect.objectContaining(expectedParams));
  expect(synchronized.cv_random_state).toBe(expected);
});

/** Connected selection comparison is order-insensitive and parameter-only edits do not emit a patch. */
it('keeps the existing no-op behavior for reordered connected models and parameter-only changes', async () => {
  connectModels([{ model_type: 'logistic_regression', hyperparameters: { C: 2 } }, { model_type: 'random_forest_classifier' }]);
  const onChange = await renderSettings();
  expect(onChange).not.toHaveBeenCalled();
});

/** A manual task locks out both connected-task and target-dtype inference. */
it('honors a manual task with incompatible connected models and a floating target', async () => {
  connectModels([{ model_type: 'ridge_regression' }]);
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ availableColumns: [column('float64')] }));
  const onChange = await renderSettings({ task_manual: true });
  expect(onChange).not.toHaveBeenCalled();
});

/** Optional boosters remain selectable when registered for this task or already configured. */
it('filters optional boosters by task while retaining selected and final estimators', async () => {
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([{ id: 'xgboost_regressor', name: '', category: '', description: '', params: {} }]);
  const { unmount } = render(<EnsembleSettings config={config} onChange={vi.fn()} nodeId="ensemble" />);
  await act(async () => {});
  expect(screen.queryByRole('button', { name: 'XGBoost' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'LightGBM' })).not.toBeInTheDocument();
  unmount();
  await renderSettings({ base_estimators: ['random_forest', 'xgboost'], final_estimator: 'lightgbm' });
  expect(screen.getByRole('button', { name: 'XGBoost' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByRole('button', { name: 'LightGBM' })).toHaveAttribute('aria-pressed', 'false');
});

/** Backend availability exposes only boosters matching the current ensemble task. */
it.each(['classification', 'regression'] as const)('offers registered boosters for %s', async (task) => {
  const suffix = task === 'classification' ? 'classifier' : 'regressor';
  vi.mocked(registryApi.getAllNodes).mockResolvedValue([
    { id: `xgboost_${suffix}`, name: '', category: '', description: '', params: {} },
    { id: `lgbm_${suffix}`, name: '', category: '', description: '', params: {} },
  ]);
  await renderSettings({ task });
  expect(screen.getByRole('button', { name: 'XGBoost' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'LightGBM' })).toBeVisible();
});

/** Upstream target filling must not overwrite an explicit local target choice. */
it.each(['', 'local'])('fills the upstream target only when the local target is "%s"', async (target_column) => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ upstreamTarget: 'upstream' }));
  const onChange = await renderSettings({ target_column });
  if (target_column) {
    expect(onChange).not.toHaveBeenCalled();
  } else {
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ ...config, target_column: 'upstream' });
  }
});

/** Zero values and checkbox false must survive the real conditional settings controls. */
it('preserves numeric zero values and emits calibration, CV and tuning edits', async () => {
  const onChange = await renderSettings({ run_mode: 'advanced', weights: { random_forest: 0 }, calibrate_base_models: true,
    calibration_cv: 0, cv_enabled: true, cv_random_state: 0, random_state: 0, strategy_params: { stale: true } });
  expect(screen.getByRole('spinbutton', { name: 'Random Forest weight' })).toHaveValue(0);
  expect(screen.getByRole('spinbutton', { name: 'Calibration CV Folds' })).toHaveValue(0);
  expect(screen.getByRole('spinbutton', { name: 'Random State' })).toHaveValue(0);
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Logistic Regression weight' }), { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ weights: { random_forest: 0, logistic_regression: 0 } }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Calibration Method' }), { target: { value: 'isotonic' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ calibration_method: 'isotonic' }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Search Strategy' }), { target: { value: 'optuna' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ search_strategy: 'optuna', strategy_params: {} }));
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  expect(screen.getByRole('spinbutton', { name: 'Fold Split Seed' })).toHaveValue(0);
  fireEvent.click(screen.getByRole('checkbox', { name: 'Shuffle Data' }));
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ cv_shuffle: false }));
});

/** CV lists time columns first and hides the split seed only for time-series methods. */
it('groups time columns without reordering each group', async () => {
  vi.mocked(useTrainingNodeContext).mockReturnValue(trainingContext({ availableColumns: [
    { ...column('float64'), name: 'amount' }, { ...column('datetime64'), name: 'when' },
    { ...column('string'), name: 'category' }, { ...column('date'), name: 'day' },
  ] }));
  await renderSettings({ task_manual: true, cv_enabled: true, cv_type: 'time_series_split' });
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  const options = within(screen.getByRole('combobox', { name: 'Time Column (optional)' })).getAllByRole('option');
  expect(options.map((option) => option.textContent)).toEqual(['Auto-detect', 'when', 'day', 'amount', 'category']);
  expect(screen.queryByRole('spinbutton', { name: 'Fold Split Seed' })).not.toBeInTheDocument();
});

/** Parent-owned CV expansion survives mode changes while advanced details reset when unmounted. */
it('preserves the existing settings lifetimes across mode changes', async () => {
  function Harness() {
    const [current, setCurrent] = useState(config);
    return <EnsembleSettings config={current} onChange={setCurrent} nodeId="ensemble" />;
  }
  await act(async () => { render(<Harness />); });
  fireEvent.click(screen.getByRole('button', { name: 'Cross Validation' }));
  fireEvent.click(screen.getByRole('button', { name: 'Advanced (Tuning)' }));
  fireEvent.click(screen.getByRole('button', { name: 'Base model tuning details' }));
  expect(screen.getByRole('button', { name: 'Base model tuning details' })).toHaveAttribute('aria-expanded', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Basic' }));
  fireEvent.click(screen.getByRole('button', { name: 'Advanced (Tuning)' }));
  expect(screen.getByRole('button', { name: 'Base model tuning details' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('button', { name: 'Cross Validation' })).toHaveAttribute('aria-expanded', 'true');
});
