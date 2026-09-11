import { test, expect } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

test('correlation selection previews without a target and preserves supervised validation', async ({ page }) => {
  // Real method changes must unblock target-free Preview while supervised methods stay guarded.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: {
      a: { name: 'a', dtype: 'float64' }, b: { name: 'b', dtype: 'float64' },
      other: { name: 'other', dtype: 'float64' },
    }, row_count: 4,
  } }));
  await page.route('**/api/pipeline/preview?*', route => route.fulfill({ json: {
    pipeline_id: 'oc31', status: 'success', recommendations: [],
    node_results: { selection: { status: 'success', metrics: { dropped_columns: ['b'] } } },
    preview_data: [{ a: 1, other: 1 }],
  } }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } };
    }).__skyulfTest.graphStore.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([
      {
        id: 'dataset', type: 'custom', position: { x: 0, y: 100 },
        data: { definitionType: 'dataset_node', datasetId: 'iris-demo', datasetName: 'Iris (demo)' },
      },
      {
        id: 'selection', type: 'custom', position: { x: 350, y: 100 }, selected: true,
        data: {
          definitionType: 'feature_selection', label: 'Feature Selection',
          method: 'select_k_best', k: 1, threshold: 0.9, correlation_method: 'pearson',
        },
      },
    ], [{ id: 'edge', source: 'dataset', target: 'selection', sourceHandle: 'data', targetHandle: 'in' }]);
  });

  const issue = page.getByRole('button', { name: /Configuration Feature Selection.*Target column is required/i });
  const method = page.getByRole('combobox', { name: 'Selection Method', exact: true });
  const target = page.getByRole('combobox', { name: 'Target Column', exact: true });
  await expect(issue).toBeVisible();
  await expect(target).toBeVisible();
  await method.press('Home');
  await method.press('ArrowDown');
  await method.press('Enter');
  await expect(method).toHaveValue('correlation_threshold');
  await expect(issue).toHaveCount(0);
  await expect(target).toHaveCount(0);

  const pendingRequest = page.waitForRequest(request =>
    request.method() === 'POST' && request.url().includes('/api/pipeline/preview?'));
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const request = await pendingRequest;
  const selection = request.postDataJSON().nodes.find((node: { node_id: string }) => node.node_id === 'selection');
  expect(selection.step_type).toBe('feature_selection');
  expect(selection.params).toEqual(expect.objectContaining({
    method: 'correlation_threshold', threshold: 0.9, correlation_method: 'pearson',
  }));
  expect(selection.params).not.toHaveProperty('target_column');
  await expect(page.getByRole('columnheader', { name: 'a', exact: true })).toBeVisible();
  await expect(page.getByRole('columnheader', { name: 'b', exact: true })).toHaveCount(0);

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(method).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(method).toHaveValue('correlation_threshold');
  await expect(issue).toHaveCount(0);
  await method.selectOption('select_k_best');
  await expect(target).toBeVisible();
  await expect(issue).toBeVisible();
});
