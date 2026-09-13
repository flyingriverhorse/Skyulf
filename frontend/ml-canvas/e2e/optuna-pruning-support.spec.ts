import { expect, test, type Page, type Route } from '@playwright/test';
import type { PipelineConfigModel } from '../src/core/api/client';
import { mockBackend } from './fixtures/mockApi';
import { pruningSearchSpaces } from './fixtures/pruningSearchSpaces';

interface SupportRequest {
  model_type: string;
  search_space: Record<string, unknown[]>;
  pipeline: PipelineConfigModel;
  node_id: string;
  strategy_params: Record<string, unknown>;
}

const FOREST_REASON = 'Random Forest requires at least two CV folds for pruning; a single holdout cannot stop trials early.';
const FOLD_REASON = 'A single holdout offers no between-fold pruning for this model and preprocessing.';
const FOLD_SUPPORT = { supported: true, mode: 'folds', reason: 'Early stopping is available between CV folds.' };
const ITERATION_SUPPORT = { supported: true, mode: 'iterations', reason: 'Early stopping is available during training.' };
const SAVED_STRATEGY = { pruner: 'hyperband', sampler: 'random', timeout: 73 };

/** Keep saved search defaults valid for each algorithm without replacing Canvas settings. */
function searchSpace(model: string): Record<string, unknown[]> {
  return pruningSearchSpaces[model] ?? {};
}

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [
    { id: 'random_forest_classifier', name: 'Random Forest', tags: ['classification'] },
    { id: 'sgd_classifier', name: 'SGD Classifier', tags: ['classification', 'text'] },
    { id: 'ridge_regression', name: 'Ridge Regression', tags: ['regression'] },
    { id: 'xgboost_classifier', name: 'XGBoost Classifier', tags: ['classification'] },
    { id: 'lgbm_classifier', name: 'LightGBM Classifier', tags: ['classification'] },
  ].map(model => ({ ...model, category: 'Modeling', description: '', params: {} })) }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*/defaults*', route => {
    const model = new URL(route.request().url()).pathname.split('/').at(-2)!;
    return route.fulfill({ json: searchSpace(model) });
  });
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    x: { name: 'x', dtype: 'float64' }, target: { name: 'target', dtype: 'int64', unique_count: 2 },
  } } }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [] }));
});

/** Seed a saved graph through real graph actions and wait for ordinary settings initialization. */
async function prepareModel(page: Page, model = 'random_forest_classifier', type = 'classification', pruner = 'hyperband', cvEnabled = model !== 'random_forest_classifier') {
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const ids = await page.evaluate(({ model, type, strategy, cvEnabled }) => {
    const store = window.__skyulfTest!.graphStore;
    const state = store.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 }, { datasetId: 'pruning-data', datasetName: 'Pruning data' });
    const modelId = state.addNode(type, { x: 260, y: 0 }, {
      run_mode: 'advanced', model_type: model, target_column: 'target', search_strategy: 'optuna',
      strategy_params: strategy, search_space: {}, cv_enabled: cvEnabled, cv_folds: 3, cv_type: 'k_fold',
    });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: modelId, targetHandle: 'in' });
    return { dataset, model: modelId };
  }, { model, type, strategy: { ...SAVED_STRATEGY, pruner }, cvEnabled });
  await expect(page.getByRole('button', { name: 'Search strategy settings', exact: true })).toBeVisible();
  if (type !== 'EnsembleNode') {
    await expect.poll(() => page.evaluate(id => window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === id)!.data.search_space, ids.model))
      .toEqual(searchSpace(model));
  }
  await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().clear());
  return ids;
}

/** Observe persisted config and undo history independently from the modal's local draft. */
async function savedState(page: Page, id: string) {
  return page.evaluate(id => {
    const store = window.__skyulfTest!.graphStore;
    return { config: store.getState().nodes.find(node => node.id === id)!.data.strategy_params,
      past: store.temporal.getState().pastStates.length, future: store.temporal.getState().futureStates.length };
  }, id);
}

/** Enter Optuna through the same named action used by keyboard and pointer users. */
async function openSettings(page: Page) {
  await page.getByRole('button', { name: 'Search strategy settings', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Optuna Settings', exact: true });
  await expect(dialog).toBeVisible();
  return dialog;
}

test('unsupported saved Hyperband remains unchanged until Apply and submits None with other settings', async ({ page }) => {
  // Merely inspecting an unsupported saved choice must not mutate graph state or undo history.
  const requests: SupportRequest[] = [];
  await page.route('**/api/pipeline/pruning-support', route => {
    expect(route.request().method()).toBe('POST');
    requests.push(route.request().postDataJSON() as SupportRequest);
    return route.fulfill({ json: { supported: false, mode: 'none', reason: FOREST_REASON } });
  });
  const ids = await prepareModel(page);
  const before = await savedState(page, ids.model);
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('none');
  await expect(pruner).toHaveAccessibleDescription(FOREST_REASON);
  await expect(dialog.getByText(FOREST_REASON, { exact: true })).toBeVisible();
  expect(requests.at(-1)).toMatchObject({ model_type: 'random_forest_classifier', node_id: ids.model,
    search_space: searchSpace('random_forest_classifier'), strategy_params: SAVED_STRATEGY });
  expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)).toMatchObject({
    step_type: 'training', inputs: [ids.dataset], params: { algorithm: 'random_forest_classifier', run_mode: 'tuned' },
  });
  expect(await savedState(page, ids.model)).toEqual(before);
  await dialog.press('Escape');
  expect(await savedState(page, ids.model)).toEqual(before);
  await openSettings(page);
  await expect(pruner).toHaveValue('none');
  await dialog.getByRole('button', { name: 'Apply Settings', exact: true }).click();
  expect(await savedState(page, ids.model)).toEqual({ config: { ...SAVED_STRATEGY, pruner: 'none' }, past: 1, future: 0 });

  let submission: (PipelineConfigModel & { target_node_id: string; job_type: string }) | undefined;
  await page.route('**/api/pipeline/run', route => {
    submission = route.request().postDataJSON() as typeof submission;
    return route.fulfill({ json: { job_id: 'pruning-run', job_ids: ['pruning-run'], pipeline_id: 'pruning-pipeline', message: 'Submitted' } });
  });
  await page.getByRole('button', { name: 'Tune model', exact: true }).click();
  await expect.poll(() => submission?.target_node_id).toBe(ids.model);
  expect(submission?.job_type).toBe('tuning');
  expect(submission!.nodes.find(node => node.node_id === ids.model)!.params).toMatchObject({
    tuning_config: { strategy: 'optuna', strategy_params: { pruner: 'none', sampler: 'random', timeout: 73 } },
  });
});

for (const [type, model, choice] of [
  ['classification', 'sgd_classifier', 'median'],
  ['regression', 'ridge_regression', 'hyperband'],
  ['text_classification', 'sgd_classifier', 'hyperband'],
]) {
  test(`${type} sends its model, node and search space while retaining supported ${choice}`, async ({ page }) => {
    // All supervised inspectors must wire their actual node context into the shared support check.
    const requests: SupportRequest[] = [];
    await page.route('**/api/pipeline/pruning-support', route => {
      requests.push(route.request().postDataJSON() as SupportRequest);
      return route.fulfill({ json: FOLD_SUPPORT });
    });
    const ids = await prepareModel(page, model, type, choice);
    const before = await savedState(page, ids.model);
    const dialog = await openSettings(page);
    const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
    await expect.poll(() => requests.length).toBeGreaterThan(0);
    await expect(pruner).toBeEnabled();
    await expect(pruner).toHaveValue(choice!);
    expect(requests.at(-1)).toMatchObject({ model_type: model, node_id: ids.model,
      search_space: searchSpace(model!), strategy_params: { ...SAVED_STRATEGY, pruner: choice } });
    expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)).toMatchObject({
      step_type: 'training', inputs: [ids.dataset], params: { algorithm: model, run_mode: 'tuned' },
    });
    expect(await savedState(page, ids.model)).toEqual(before);
  });
}

/** Connect a real scaling branch while keeping the active model's inspector mounted. */
async function addScalingBranch(page: Page, ids: { dataset: string; model: string }) {
  return page.evaluate(ids => {
    const state = window.__skyulfTest!.graphStore.getState();
    const scaling = state.addNode('scale_numeric_features', { x: 120, y: 200 }, { columns: ['x'], method: 'standard' });
    state.onConnect({ source: ids.dataset, sourceHandle: 'data', target: scaling, targetHandle: 'in' });
    state.selectNode(ids.model);
    return scaling;
  }, ids);
}

/** Rewire the selected model through real edge actions without dismissing its local settings draft. */
async function replaceModelInput(page: Page, model: string, source: string, sourceHandle: string) {
  await page.evaluate(({ model, source, sourceHandle }) => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.onEdgesChange(state.edges.filter(edge => edge.target === model).map(edge => ({ id: edge.id, type: 'remove' })));
    state.onConnect({ source, sourceHandle, target: model, targetHandle: 'in' });
  }, { model, source, sourceHandle });
}

test('Random Forest keeps fold pruning with scaling and CV changes restore its unapplied choice', async ({ page }) => {
  // Preprocessing remains eligible, while holdout changes must not erase a local pruner draft.
  const requests: SupportRequest[] = [];
  await page.route('**/api/pipeline/pruning-support', route => {
    const request = route.request().postDataJSON() as SupportRequest;
    requests.push(request);
    const target = request.pipeline.nodes.find(node => node.node_id === request.node_id)!;
    const tuning = target.params.tuning_config as { cv_enabled: boolean };
    return route.fulfill({ json: tuning.cv_enabled ? FOLD_SUPPORT : { supported: false, mode: 'none', reason: FOLD_REASON } });
  });
  const ids = await prepareModel(page, 'random_forest_classifier', 'classification', 'hyperband', true);
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect(pruner).toBeEnabled();
  await pruner.selectOption('median');
  const scaling = await addScalingBranch(page, ids);
  await expect.poll(() => requests.at(-1)?.pipeline.nodes.some(node => node.node_id === scaling)).toBe(true);
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveValue('median');
  expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)!.inputs).toEqual([ids.dataset]);

  await replaceModelInput(page, ids.model, scaling, 'out');
  await expect.poll(() => requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)!.inputs).toEqual([scaling]);
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveAccessibleDescription(/between CV folds/);
  await page.evaluate(id => window.__skyulfTest!.graphStore.getState().updateNodeData(id, { cv_enabled: false }), ids.model);
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('none');
  await expect(pruner).toHaveAccessibleDescription(FOLD_REASON);
  expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)!.inputs).toEqual([scaling]);
  const beforeRecovery = await savedState(page, ids.model);
  expect(beforeRecovery.config).toEqual(SAVED_STRATEGY);
  await page.evaluate(id => window.__skyulfTest!.graphStore.getState().updateNodeData(id, { cv_enabled: true }), ids.model);
  const afterRewire = await savedState(page, ids.model);
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveValue('median');
  expect(await savedState(page, ids.model)).toEqual(afterRewire);
  await dialog.getByRole('button', { name: 'Apply Settings', exact: true }).click();
  expect(await savedState(page, ids.model)).toEqual({ ...afterRewire,
    config: { ...SAVED_STRATEGY, pruner: 'median' }, past: afterRewire.past + 1 });
});

for (const model of ['xgboost_classifier', 'lgbm_classifier']) {
  test(`${model} keeps pruning during training with a real scaling node and holdout`, async ({ page }) => {
    // Boosting callbacks support fold preprocessing even when only one validation split is used.
    const requests: SupportRequest[] = [];
    await page.route('**/api/pipeline/pruning-support', route => {
      requests.push(route.request().postDataJSON() as SupportRequest);
      return route.fulfill({ json: ITERATION_SUPPORT });
    });
    const ids = await prepareModel(page, model, 'classification', 'hyperband', false);
    const scaling = await addScalingBranch(page, ids);
    await replaceModelInput(page, ids.model, scaling, 'out');
    const before = await savedState(page, ids.model);
    const dialog = await openSettings(page);
    const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
    await expect(pruner).toBeEnabled();
    await expect(pruner).toHaveValue('hyperband');
    await expect(pruner).toHaveAccessibleDescription(/during training/);
    expect(requests.at(-1)).toMatchObject({ model_type: model, search_space: searchSpace(model) });
    expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)).toMatchObject({
      inputs: [scaling], params: { tuning_config: { cv_enabled: false, cv_folds: 3 } },
    });
    await dialog.getByRole('button', { name: 'Models with pruning support', exact: true }).focus();
    const tooltip = page.getByRole('tooltip');
    await expect(tooltip).toContainText('XGBoost and LightGBM');
    await expect(tooltip).toContainText('Random Forest, Logistic Regression, SVC');
    await expect(tooltip).toContainText('single holdout');
    await tooltip.press('Escape');
    await expect(dialog).toBeVisible();
    expect(await savedState(page, ids.model)).toEqual(before);
  });
}

test('late unsupported response cannot replace a reopened supported model selection', async ({ page }) => {
  // A closed model's HTTP response must not disable the next model or silently alter saved choices.
  const pending: Route[] = [];
  const requests: SupportRequest[] = [];
  await page.route('**/api/pipeline/pruning-support', route => {
    const request = route.request().postDataJSON() as SupportRequest;
    requests.push(request);
    if (request.model_type === 'random_forest_classifier') { pending.push(route); return; }
    return route.fulfill({ json: ITERATION_SUPPORT });
  });
  const ids = await prepareModel(page);
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect.poll(() => pending.length).toBeGreaterThan(0);
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('hyperband');
  await dialog.press('Escape');
  await page.getByRole('combobox', { name: 'Model Type', exact: true }).selectOption('xgboost_classifier');
  await expect.poll(() => page.evaluate(id => window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === id)!.data.search_space, ids.model))
    .toEqual(searchSpace('xgboost_classifier'));
  const before = await savedState(page, ids.model);
  await openSettings(page);
  await expect(pruner).toBeEnabled();
  expect(requests.at(-1)?.model_type).toBe('xgboost_classifier');
  await Promise.all(pending.map(route => route.fulfill({ json: { supported: false, mode: 'none', reason: FOREST_REASON } })));
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(dialog.getByText(FOREST_REASON, { exact: true })).toHaveCount(0);
  expect(await savedState(page, ids.model)).toEqual(before);
});

test('restored graph waits for its new check and ignores the previous graph response', async ({ page }) => {
  // Revisiting an earlier graph cannot reuse stale capability while its replacement request is pending.
  const pending: Route[] = [];
  await page.route('**/api/pipeline/pruning-support', route => { pending.push(route); });
  const ids = await prepareModel(page, 'sgd_classifier');
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect.poll(() => pending.length).toBe(1);
  await pending[0]!.fulfill({ json: FOLD_SUPPORT });
  await expect(pruner).toBeEnabled();
  const scaling = await addScalingBranch(page, ids);
  await expect.poll(() => pending.length).toBe(2);
  await pending[1]!.fulfill({ json: FOLD_SUPPORT });
  await expect(pruner).toBeEnabled();
  await replaceModelInput(page, ids.model, scaling, 'out');
  await expect.poll(() => pending.length).toBe(3);
  await expect(pruner).toBeDisabled();
  await page.evaluate(() => window.__skyulfTest!.graphStore.temporal.getState().undo(2));
  await expect.poll(() => pending.length).toBe(4);
  const before = await savedState(page, ids.model);
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(pruner).toHaveAccessibleDescription(/checking/i);
  await pending[3]!.fulfill({ json: FOLD_SUPPORT });
  await expect(pruner).toBeEnabled();
  await pending[2]!.fulfill({ json: { supported: false, mode: 'none', reason: FOLD_REASON } });
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(dialog.getByText(FOLD_REASON, { exact: true })).toHaveCount(0);
  expect(await savedState(page, ids.model)).toEqual(before);
});

test('pending and failed support checks keep saved Hyperband while disabling only selection', async ({ page }) => {
  // An unavailable capability endpoint cannot erase the persisted choice or invent a supported result.
  const pending: Route[] = [];
  await page.route('**/api/pipeline/pruning-support', route => { pending.push(route); });
  const ids = await prepareModel(page, 'sgd_classifier');
  const before = await savedState(page, ids.model);
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect.poll(() => pending.length).toBeGreaterThan(0);
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(pruner).toHaveAccessibleDescription(/checking/i);
  expect(await savedState(page, ids.model)).toEqual(before);
  await Promise.all(pending.map(route => route.fulfill({ status: 503, json: { detail: 'Service temporarily unavailable.' } })));
  await expect(pruner).toHaveAccessibleDescription(/could not|unable|failed/i);
  await expect(pruner).toBeDisabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(dialog.getByRole('combobox', { name: 'Sampler', exact: true })).toHaveValue('random');
  await expect(dialog.getByRole('spinbutton', { name: 'Timeout (Seconds)', exact: true })).toHaveValue('73');
  await dialog.press('Escape');
  expect(await savedState(page, ids.model)).toEqual(before);
});

test('ensemble settings retain Hyperband for supported CV without rewriting saved settings', async ({ page }) => {
  // The ensemble consumer must pass its own model and node context to the shared capability UI.
  const requests: SupportRequest[] = [];
  await page.route('**/api/pipeline/pruning-support', route => {
    requests.push(route.request().postDataJSON() as SupportRequest);
    return route.fulfill({ json: FOLD_SUPPORT });
  });
  const ids = await prepareModel(page, 'voting_classifier', 'EnsembleNode');
  const before = await savedState(page, ids.model);
  const dialog = await openSettings(page);
  const pruner = dialog.getByRole('combobox', { name: 'Pruner', exact: true });
  await expect(pruner).toBeEnabled();
  await expect(pruner).toHaveValue('hyperband');
  await expect(pruner).toHaveAccessibleDescription(/between CV folds/);
  expect(requests.at(-1)).toMatchObject({ model_type: 'voting_classifier', node_id: ids.model, strategy_params: SAVED_STRATEGY });
  expect(requests.at(-1)!.pipeline.nodes.find(node => node.node_id === ids.model)).toMatchObject({
    params: { tuning_config: { cv_enabled: true, cv_folds: 3 } },
  });
  expect(await savedState(page, ids.model)).toEqual(before);
});
