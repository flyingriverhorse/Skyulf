import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test.beforeEach(async ({ page }) => {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    x: { name: 'x', dtype: 'float64' }, z: { name: 'z', dtype: 'float64' }, target: { name: 'target', dtype: 'float64' },
  } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
});

test('joint filters and percentile overrides keep keyboard validation and mobile state', async ({ page }) => {
  // Actual settings must send the joint owner and surface invalid per-column tails accessibly.
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'numeric', datasetName: 'Numeric' });
    const id = state.addNode('outlier', { x: 250, y: 100 }, { method: 'winsorize', columns: ['x', 'z'] });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const method = page.getByRole('combobox', { name: 'Method', exact: true });
  const override = page.getByRole('checkbox', { name: 'Override percentiles for x' });
  await override.focus();
  await override.press('Space');
  const lower = page.getByRole('spinbutton', { name: 'x lower percentile' });
  const upper = page.getByRole('spinbutton', { name: 'x upper percentile' });
  await lower.fill('90');
  await upper.fill('10');
  const issue = page.getByRole('button', { name: /Configuration Outlier Removal.*Upper percentile/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(upper).toBeFocused();
  await expect(upper).toHaveAccessibleDescription(/Upper percentile/);
  await upper.fill('99');
  await method.selectOption('local_outlier_factor');
  await page.getByRole('spinbutton', { name: 'Neighbors' }).fill('8');
  let payload: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    payload = route.request().postDataJSON() as typeof payload;
    return route.fulfill({ json: { pipeline_id: 'joint', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => payload?.nodes.find(node => node.node_id === id)?.step_type).toBe('LocalOutlierFactor');
  expect(payload!.nodes.find(node => node.node_id === id)?.params).toMatchObject({ columns: ['x', 'z'], n_neighbors: 8 });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page.getByRole('spinbutton', { name: 'Neighbors' })).toHaveValue('8');
  await method.selectOption('winsorize');
  await expect(override).toBeChecked();
  await expect(lower).toHaveValue('90');
});

test('cross-validated feature selection exposes actionable fold errors and exact preview settings', async ({ page }) => {
  // The existing selection node must retain separate sequential count and CV controls in its payload.
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'numeric', datasetName: 'Numeric' });
    const id = state.addNode('feature_selection', { x: 250, y: 100 }, { method: 'rfecv', target_column: 'target', cv: 3 });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const cv = page.getByRole('spinbutton', { name: 'CV folds' });
  await cv.fill('1');
  const issue = page.getByRole('button', { name: /Configuration Feature Selection.*cv/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(cv).toBeFocused();
  await cv.fill('3');
  await page.getByRole('combobox', { name: 'Selection Method' }).selectOption('sequential_feature_selection');
  await page.getByRole('combobox', { name: 'Direction' }).selectOption('backward');
  await page.getByRole('textbox', { name: 'Features to select' }).fill('2');
  let payload: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    payload = route.request().postDataJSON() as typeof payload;
    return route.fulfill({ json: { pipeline_id: 'selection', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => payload?.nodes.find(node => node.node_id === id)?.step_type).toBe('feature_selection');
  expect(payload!.nodes.find(node => node.node_id === id)?.params).toMatchObject({ method: 'sequential_feature_selection', cv: 3, n_features_to_select: 2, direction: 'backward' });
});
