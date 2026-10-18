import { test, expect } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('constant imputation keeps exact text in the real preview payload', async ({ page }) => {
  // JavaScript number conversion must not round integer text before Core validates each column.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: { value: { name: 'value', dtype: 'float64' } } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'constants', datasetName: 'Constants' });
    const id = state.addNode('imputation_node', { x: 250, y: 100 }, { method: 'simple', columns: ['value'], strategy: 'constant', fill_value: 0 });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  const input = page.getByRole('textbox', { name: 'Fill Value' });
  await expect(input).toHaveValue('0');
  await input.fill('9007199254740993');
  let submitted: { nodes: { node_id: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'constant-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  expect(submitted!.nodes.find(node => node.node_id === id)!.params.fill_value).toBe('9007199254740993');
});

test('ordered group controls retain key order through keyboard, preview and mobile layout', async ({ page }) => {
  // Real controls must serialize their displayed fallback order and retain legacy opt-in behavior.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: { columns: {
    region: { name: 'region', dtype: 'object' }, industry: { name: 'industry', dtype: 'object' }, value: { name: 'value', dtype: 'float64' },
  } } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  const id = await page.evaluate(() => {
    const state = window.__skyulfTest!.graphStore.getState();
    state.setGraph([], []);
    const source = state.addNode('dataset_node', { x: 0, y: 100 }, { datasetId: 'groups', datasetName: 'Groups' });
    const id = state.addNode('imputation_node', { x: 250, y: 100 }, { method: 'group', columns: ['value'], group_by: 'region', strategy: 'mean' });
    state.onConnect({ source, sourceHandle: 'data', target: id, targetHandle: 'in' });
    return id;
  });
  await expect(page.getByRole('combobox', { name: 'Group By', exact: true })).toHaveValue('region');
  const multiple = page.getByRole('checkbox', { name: 'Use multiple group columns' });
  await multiple.focus();
  await multiple.press('Space');
  const groups = page.getByRole('group', { name: 'Ordered Group Columns' });
  await groups.getByRole('checkbox', { name: 'industry', exact: true }).check();
  await page.getByRole('combobox', { name: 'Group Fallback' }).selectOption('hierarchical');
  await page.getByRole('button', { name: 'Move industry earlier' }).click();
  await expect(page.getByRole('list', { name: 'Group fallback order' }).getByRole('listitem').first()).toContainText('1. industry');
  let submitted: { nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[] } | undefined;
  await page.route('**/api/pipeline/preview?*', route => {
    submitted = route.request().postDataJSON() as typeof submitted;
    return route.fulfill({ json: { pipeline_id: 'groups-preview', status: 'success', node_results: {}, preview_data: {}, recommendations: [] } });
  });
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect.poll(() => submitted?.nodes.length).toBe(2);
  expect(submitted!.nodes.find(node => node.node_id === id)).toMatchObject({ step_type: 'GroupImputer', params: { group_by: ['industry', 'region'], fallback: 'hierarchical' } });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(multiple).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await expect(page.getByRole('combobox', { name: 'Group Fallback' })).toHaveValue('hierarchical');
  await expect(page.getByRole('list', { name: 'Group fallback order' }).getByRole('listitem').first()).toContainText('1. industry');
});
