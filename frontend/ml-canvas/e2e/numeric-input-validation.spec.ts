import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('a saved invalid search draft stays blocked after reload and an intentional None edit repairs it', async ({ page }) => {
  // JSON persistence must not reinterpret an overflow draft as a valid null candidate.
  await mockBackend(page);
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [{
    id: 'random_forest_classifier', name: 'Random Forest', category: 'Modeling', description: '', params: {}, tags: ['classification'],
  }] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [{ name: 'max_depth', label: 'Max Depth', type: 'number', default: null }] }));
  await page.route('**/api/pipeline/hyperparameters/*/defaults*', route => route.fulfill({ json: {} }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    x: { name: 'x', dtype: 'float64' }, target: { name: 'target', dtype: 'int64' },
  } } }));
  let submissions = 0;
  await page.route('**/api/pipeline/run', route => { submissions += 1; return route.fulfill({ json: {} }); });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 }, { datasetId: 'data', datasetName: 'Data' });
    const config = JSON.parse(JSON.stringify({ run_mode: 'advanced', model_type: 'random_forest_classifier', target_column: 'target',
      search_space: { max_depth: [null] }, invalid_search_space: { max_depth: '1e400' }, n_trials: 2, random_state: 0 }));
    const model = state.addNode('classification', { x: 260, y: 0 }, config);
    state.onConnect({ source: dataset, sourceHandle: 'data', target: model, targetHandle: 'in' });
  });
  const run = page.getByRole('button', { name: 'Tune model', exact: true });
  await expect(run).toBeDisabled();
  await page.getByRole('button', { name: /Configuration Classification.*numeric candidates must be finite/i }).click();
  const input = page.getByRole('textbox', { name: 'Max Depth', exact: true });
  await expect(input).toHaveValue('1e400');
  await expect(input).toHaveAttribute('aria-invalid', 'true');
  await input.fill('None');
  await input.blur();
  await expect(run).toBeEnabled();
  expect(submissions).toBe(0);
});

test('clearing a real split ratio blocks execution and its issue focuses the invalid field', async ({ page }) => {
  // The browser's empty numeric input must never become a silently valid split.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/jobs?*', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/list', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: { x: { name: 'x', dtype: 'float64' } } } }));
  let submissions = 0;
  await page.route('**/api/pipeline/run', route => { submissions += 1; return route.fulfill({ json: {} }); });
  await page.route('**/api/pipeline/preview', route => { submissions += 1; return route.fulfill({ json: {} }); });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 0 }, { datasetId: 'data', datasetName: 'Data' });
    const split = state.addNode('TrainTestSplitter', { x: 260, y: 0 });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: split, targetHandle: 'in' });
  });
  const input = page.getByRole('spinbutton', { name: 'Validation Size (0.0 - 1.0)', exact: true });
  await input.fill('');
  const issue = page.getByRole('button', { name: /Configuration Train-Test Split.*Validation size must be between 0 and 1/i });
  await expect(issue).toBeVisible();
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  expect(submissions).toBe(0);
  await issue.click();
  await expect(input).toBeFocused();
  await expect(input).toHaveAttribute('aria-invalid', 'true');
  await input.fill('0');
  await expect(issue).toHaveCount(0);
  await expect(input).not.toHaveAttribute('aria-invalid');
  expect(submissions).toBe(0);
});
