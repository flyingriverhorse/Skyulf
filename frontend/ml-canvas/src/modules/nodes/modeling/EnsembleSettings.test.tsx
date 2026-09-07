import { act, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { registryApi } from '../../../core/api/registry';
import { useTrainingNodeContext } from '../../../core/hooks/useTrainingNodeContext';
import { useGraphStore } from '../../../core/store/useGraphStore';
import { useJobStore } from '../../../core/store/useJobStore';
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
  const onChange = vi.fn();
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
