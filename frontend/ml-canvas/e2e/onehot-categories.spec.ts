import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('fixed categories support keyboard editing, saved previews and mobile read-only mode', async ({ page }) => {
  // Real settings must retain exact category text and remain safe across the mobile breakpoint.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    city: { name: 'city', dtype: 'object' },
  } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const dataset = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'cities', datasetName: 'Cities' });
    const id = state.addNode('encoding', { x: 250, y: 100 }, { columns: ['city'] });
    state.onConnect({ source: dataset, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const fixed = page.getByRole('checkbox', { name: 'Use fixed categories' });
  await fixed.focus();
  await fixed.press('Space');
  const levels = page.getByRole('textbox', { name: 'Categories for city' });
  const issue = page.getByRole('button', { name: /Configuration Encoding.*unique categories/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(levels).toBeFocused();
  await expect(levels).toHaveAccessibleDescription(/unique categories/);
  await levels.fill('Øl\nÅben\nA, B');
  await expect(issue).toHaveCount(0);
  await page.getByRole('combobox', { name: 'Column names' }).selectOption('slug');
  await expect(page.getByRole('spinbutton', { name: 'Max Categories' })).toBeDisabled();
  await expect(page.getByRole('checkbox', { name: 'Include missing as category' })).toBeDisabled();
  let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'onehot-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  const params = submitted!.nodes.find(node => node.node_id === id)!.params;
  expect(params).toMatchObject({ categories: { city: ['Øl', 'Åben', 'A, B'] }, max_categories: null, include_missing: false, column_names: 'slug' });
  expect(params).not.toHaveProperty('method');
  expect(params).not.toHaveProperty('n_features');
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(levels).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(levels).toHaveValue('Øl\nÅben\nA, B');
});
