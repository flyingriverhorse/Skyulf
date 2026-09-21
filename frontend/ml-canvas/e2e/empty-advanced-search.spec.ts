import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('Basic 400/7 switched to empty Advanced search blocks fifty-trial submission', async ({ page }) => {
  // A real mode switch must expose recovery guidance before a default model can run.
  await mockBackend(page);
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [{
    id: 'random_forest_classifier', name: 'Random Forest', category: 'Modeling',
    description: '', params: {}, tags: ['classification'],
  }] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*/defaults*', route => route.fulfill({ json: {} }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    x: { name: 'x', dtype: 'float64' }, target: { name: 'target', dtype: 'int64' },
  } } }));
  let submissions = 0;
  await page.route('**/api/pipeline/run', route => {
    submissions += 1;
    return route.fulfill({ json: { job_id: 'unexpected' } });
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 }, { datasetId: 'data', datasetName: 'Data' });
    const model = state.addNode('classification', { x: 260, y: 0 }, {
      run_mode: 'basic', model_type: 'random_forest_classifier', target_column: 'target',
      hyperparameters: { n_estimators: 400, max_depth: 7 }, search_space: {}, n_trials: 50, random_state: 0,
    });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: model, targetHandle: 'in' });
    return model;
  });
  await page.getByRole('button', { name: 'Advanced (Tuning)', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Tune model', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Tune model', exact: true }))
    .toHaveAccessibleDescription(/Open Search Space and configure at least one parameter/);
  const issue = page.getByRole('button', { name: /Configuration Classification.*Configure at least one search parameter/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(page.locator('[data-validation-field="search_space"]')).toBeVisible();
  await expect(page.getByText(/Basic-mode values are retained for Basic runs only/)).toBeVisible();
  expect(await page.evaluate(id => window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === id)!.data, id))
    .toMatchObject({ run_mode: 'advanced', hyperparameters: { n_estimators: 400, max_depth: 7 }, search_space: {}, n_trials: 50, random_state: 0 });
  expect(submissions).toBe(0);
});
