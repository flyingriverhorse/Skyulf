import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('category statistics controls retain keyboard edits, preview payloads and mobile state', async ({ page }) => {
  // Real Canvas settings must export exactly the selected method controls.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: { city: { name: 'city', dtype: 'object' } } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const graph = window.__skyulfTest!.graphStore.getState();
    graph.setGraph([], []);
    const dataset = graph.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'cities', datasetName: 'Cities' });
    const encoder = graph.addNode('encoding', { x: 250, y: 100 }, { columns: ['city'] });
    graph.onConnect({ source: dataset, sourceHandle: 'data', target: encoder, targetHandle: 'in' });
    return encoder;
  });
  const method = page.getByRole('combobox', { name: 'Encoding Method' });
  await method.selectOption('frequency');
  const statistic = page.getByRole('combobox', { name: 'Statistic' });
  await statistic.focus();
  await statistic.press('End');
  await statistic.press('Tab');
  await expect(statistic).toHaveValue('count');
  await method.selectOption('rare');
  await page.getByRole('spinbutton', { name: 'Minimum count' }).fill('3');
  await page.getByRole('spinbutton', { name: 'Minimum share' }).fill('0.25');
  const replacement = page.getByRole('textbox', { name: 'Replacement label' });
  await replacement.fill('');
  const issue = page.getByRole('button', { name: /Configuration Encoding.*replacement label/i });
  await expect(issue).toBeVisible();
  await issue.click();
  await expect(replacement).toBeFocused();
  await expect(replacement).toHaveAccessibleDescription(/replacement label/);
  await replacement.fill('Grouped');
  await expect(issue).toHaveCount(0);
  let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'category-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  const params = submitted!.nodes.find(node => node.node_id === id)!.params;
  expect(params).toEqual({ _display_name: 'Encoding', columns: ['city'], min_count: 3, min_share: 0.25, replacement: 'Grouped' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(replacement).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(replacement).toHaveValue('Grouped');
  await method.selectOption('target');
  await page.getByRole('checkbox', { name: 'Shuffle folds' }).uncheck();
  await expect(page.getByRole('spinbutton', { name: 'Random seed' })).toBeDisabled();
});
