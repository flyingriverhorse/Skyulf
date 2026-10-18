import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('scaling options support keyboard validation, preview and mobile persistence', async ({ page }) => {
  // The actual settings form must retain new scaler choices through payloads and responsive layout.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    x: { name: 'x', dtype: 'float64' }, z: { name: 'z', dtype: 'float64' },
  } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'numeric', datasetName: 'Numeric' });
    const id = state.addNode('scale_numeric_features', { x: 250, y: 100 }, { method: 'minmax', columns: ['x', 'z'] });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const method = page.getByRole('combobox', { name: 'Scaling Method' });
  const clipping = page.getByRole('checkbox', { name: 'Clip values to feature range' });
  await clipping.focus();
  await clipping.press('Space');
  await expect(clipping).toBeChecked();
  await method.selectOption('quantile');
  const count = page.getByRole('spinbutton', { name: 'Quantile count' });
  await count.fill('0');
  const issue = page.getByRole('button', { name: /Configuration Scaling.*positive integer/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(count).toBeFocused();
  await expect(count).toHaveAccessibleDescription(/positive integer/);
  await count.fill('20');
  await page.getByRole('combobox', { name: 'Output distribution' }).selectOption('normal');
  await expect(issue).toHaveCount(0);
  let submitted: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'scaling-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.find(node => node.node_id === id)?.step_type).toBe('QuantileTransformer');
  expect(submitted!.nodes.find(node => node.node_id === id)).toMatchObject({
    params: { columns: ['x', 'z'], output_distribution: 'normal', n_quantiles: 20 },
  });
  await method.selectOption('normalize');
  const norm = page.getByRole('combobox', { name: 'Row norm' });
  await norm.selectOption('max');
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(norm).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(norm).toHaveValue('max');
  await method.selectOption('minmax');
  await expect(clipping).toBeChecked();
});
