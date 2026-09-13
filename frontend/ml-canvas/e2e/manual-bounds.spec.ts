import { expect, test, type Page } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

/** Connect a real Outlier panel to numeric schema data without a running backend. */
async function openOutlier(page: Page) {
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: { age: { name: 'age', dtype: 'float64' }, score: { name: 'score', dtype: 'float64' } }, row_count: 4,
  } }));
  await page.route('**/api/pipeline/preview?*', route => route.fulfill({ json: {
    pipeline_id: 'manual-bounds', status: 'success', node_results: {}, recommendations: [],
    preview_data: [{ age: 0, score: 2 }],
  } }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => window.__skyulfTest!.graphStore.getState().setGraph([
    {
      id: 'dataset', type: 'custom', position: { x: 0, y: 100 },
      data: { definitionType: 'dataset_node', datasetId: 'iris-demo', datasetName: 'Iris (demo)' },
    },
    {
      id: 'outlier', type: 'custom', position: { x: 350, y: 100 }, selected: true,
      data: {
        definitionType: 'outlier', label: 'Outlier Removal', method: 'iqr', columns: ['age', 'score'],
        multiplier: 2, bounds: { age: { lower: 0 }, score: { lower: 10, upper: 20 } },
      },
    },
  ], [{ id: 'edge', source: 'dataset', target: 'outlier', sourceHandle: 'data', targetHandle: 'in' }]));
}

test('manual bounds preview only selected columns and retain edits across layouts', async ({ page }) => {
  // A removed column and inactive statistical parameters must not leak into the request.
  await openOutlier(page);
  const method = page.getByRole('combobox', { name: 'Method', exact: true });
  await method.press('End');
  await method.press('Enter');
  await expect(method).toHaveValue('manual_bounds');
  const lower = page.getByRole('spinbutton', { name: 'Lower bound for age', exact: true });
  const upper = page.getByRole('spinbutton', { name: 'Upper bound for age', exact: true });
  await expect(lower).toHaveValue('0');
  await upper.fill('5');
  await upper.fill('');
  await page.getByRole('checkbox', { name: 'score', exact: true }).press('Space');
  await expect(page.getByRole('spinbutton', { name: 'Lower bound for score', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
  await expect(lower).toHaveValue('0');
  await expect(upper).toHaveValue('');
  await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
  const pendingRequest = page.waitForRequest(request => request.method() === 'POST' && request.url().includes('/api/pipeline/preview?'));
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const request = await pendingRequest;
  const outlier = request.postDataJSON().nodes.find((node: { node_id: string }) => node.node_id === 'outlier');
  expect(outlier.step_type).toBe('ManualBounds');
  expect(outlier.params).toEqual({ bounds: { age: { lower: 0 } }, _display_name: 'Outlier Removal' });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(lower).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(lower).toHaveValue('0');
  await expect(upper).toHaveValue('');
  await method.selectOption('iqr');
  await expect(page.getByRole('spinbutton', { name: 'Multiplier', exact: true })).toHaveValue('2');
});

test('manual bounds issues focus missing and inverted endpoints and clear after correction', async ({ page }) => {
  // Issues must take keyboard users to the exact bound and recover without stealing typing focus.
  await openOutlier(page);
  await page.getByRole('combobox', { name: 'Method', exact: true }).selectOption('manual_bounds');
  const lower = page.getByRole('spinbutton', { name: 'Lower bound for age', exact: true });
  const upper = page.getByRole('spinbutton', { name: 'Upper bound for age', exact: true });
  await lower.fill('');
  const missing = page.getByRole('button', { name: /Configuration Outlier Removal.*Set a lower or upper bound for age/i });
  await expect(missing).toBeVisible();
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  await missing.press('Enter');
  await expect(lower).toBeFocused();
  await expect(lower).toHaveAccessibleDescription(/Set a lower or upper bound for age/);
  await lower.fill('10');
  await expect(missing).toHaveCount(0);
  await upper.fill('5');
  const inverted = page.getByRole('button', { name: /Configuration Outlier Removal.*Upper bound for age must be greater/i });
  await inverted.press('Enter');
  await expect(upper).toBeFocused();
  await expect(upper).toHaveAttribute('aria-invalid', 'true');
  await upper.fill('10');
  await expect(inverted).toHaveCount(0);
  await expect(upper).toBeFocused();
  await expect(upper).not.toHaveAttribute('aria-invalid');
});

test('manual bounds excludes a separated target while a stale selection remains visible and removable', async ({ page }) => {
  // Stale target rules must block preview until the user explicitly removes the unavailable column.
  await openOutlier(page);
  const previewRequests: unknown[] = [];
  page.on('request', request => {
    if (request.method() === 'POST' && request.url().includes('/api/pipeline/preview?')) previewRequests.push(request.postDataJSON());
  });
  await page.route('**/api/pipeline/schema-preview', route => {
    const operation = route.request().postDataJSON().nodes.find((node: { node_id: string }) => node.node_id === 'outlier');
    const staleTarget = operation.step_type === 'ManualBounds' && Object.hasOwn(operation.params.bounds, 'score');
    return route.fulfill({ json: {
      pipeline_id: 'manual-bounds',
      predicted_schemas: { split: { columns: ['age', 'score'], dtypes: { age: 'float64', score: 'float64' } } },
      broken_references: staleTarget ? [{ node_id: 'outlier', field: 'bounds', column: 'score', upstream_node_id: 'split' }] : [],
    } });
  });
  await page.evaluate(() => {
    const store = window.__skyulfTest!.graphStore.getState();
    store.setGraph([
      ...store.nodes.map(node => node.id === 'outlier' ? { ...node, data: { ...node.data, method: 'manual_bounds' } } : node),
      {
        id: 'split', type: 'custom', position: { x: 175, y: 100 },
        data: { definitionType: 'feature_target_split', target_column: 'score' },
      },
    ], [
      { id: 'a', source: 'dataset', target: 'split', sourceHandle: 'data', targetHandle: 'in' },
      { id: 'b', source: 'split', target: 'outlier', sourceHandle: 'X', targetHandle: 'in' },
    ]);
  });
  await expect(page.getByRole('checkbox', { name: 'score', exact: true })).toHaveCount(0);
  await expect(page.getByRole('checkbox', { name: 'age', exact: true })).toBeChecked();
  await expect(page.getByRole('spinbutton', { name: 'Upper bound for score', exact: true })).toHaveValue('20');
  const mismatch = page.getByLabel('Column mismatch: 1 column not found in upstream output', { exact: true });
  await expect(mismatch).toBeVisible();
  const issue = page.getByRole('button', { name: /configuration Outlier Removal.*score/i });
  await expect(issue).toBeVisible();
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  await expect(page.getByText('Preview blocked. Review 1 validation issue.', { exact: true })).toBeVisible();
  expect(previewRequests).toHaveLength(0);
  await page.getByRole('button', { name: 'Close settings panel', exact: true }).click();
  await issue.press('Enter');
  await expect(page.getByRole('spinbutton', { name: 'Lower bound for age', exact: true })).toBeFocused();
  expect(await page.evaluate(() => window.__skyulfTest!.graphStore.getState().nodes.find(node => node.id === 'outlier')!.data.columns))
    .toEqual(['age', 'score']);
  await page.getByRole('button', { name: 'Remove bounds for score', exact: true }).press('Enter');
  await expect(page.getByRole('spinbutton', { name: 'Upper bound for score', exact: true })).toHaveCount(0);
  await expect(issue).toHaveCount(0);
  await expect(mismatch).toHaveCount(0);
  const pendingRequest = page.waitForRequest(request => request.method() === 'POST' && request.url().includes('/api/pipeline/preview?'));
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const request = await pendingRequest;
  const operation = request.postDataJSON().nodes.find((node: { node_id: string }) => node.node_id === 'outlier');
  expect(operation.params.bounds).toEqual({ age: { lower: 0 } });
});
