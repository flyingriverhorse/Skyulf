import { test, expect } from '@playwright/test';
import { mockBackend, sampleDatasets } from './fixtures/mockApi';

test('single-column self-products can be previewed and survive mobile read-only mode', async ({ page }) => {
  // Exercise the real setting, validation and request payload, then preserve it across layouts.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: sampleDatasets } }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: { sepal_length: { name: 'sepal_length', dtype: 'float64' } }, row_count: 150,
  } }));
  await page.route('**/api/pipeline/preview?*', route => route.fulfill({ json: {
    pipeline_id: 'oc33', status: 'success', node_results: {}, recommendations: [],
    preview_data: [{ sepal_length: 2, sepal_length_x_sepal_length_x_sepal_length_x_sepal_length: 16 }],
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
        id: 'interaction', type: 'custom', position: { x: 350, y: 100 }, selected: true,
        data: {
          definitionType: 'FeatureInteractionNode', label: 'Feature Interaction',
          columns: ['sepal_length'], degree: 4, interaction_only: true, include_bias: false,
          isExpanded: true,
        },
      },
    ], [{ id: 'edge', source: 'dataset', target: 'interaction', sourceHandle: 'data', targetHandle: 'in' }]);
  });

  const issue = page.getByRole('button', { name: /Configuration Feature Interaction.*at least 4 columns/i });
  const interactionOnly = page.getByRole('checkbox', { name: 'Interaction Only', exact: true });
  await expect(issue).toBeVisible();
  await expect(interactionOnly).toBeChecked();
  await interactionOnly.press('Space');
  await expect(interactionOnly).not.toBeChecked();
  await expect(issue).toHaveCount(0);

  const pendingRequest = page.waitForRequest(request =>
    request.method() === 'POST' && request.url().includes('/api/pipeline/preview?'));
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const request = await pendingRequest;
  expect(request.postDataJSON().nodes).toEqual(expect.arrayContaining([
    expect.objectContaining({
      node_id: 'interaction', step_type: 'FeatureInteraction',
      params: expect.objectContaining({ columns: ['sepal_length'], degree: 4, interaction_only: false }),
    }),
  ]));
  await expect(page.getByRole('columnheader', {
    name: 'sepal_length_x_sepal_length_x_sepal_length_x_sepal_length', exact: true,
  })).toBeVisible();

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(interactionOnly).toHaveCount(0);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(interactionOnly).not.toBeChecked();
  await expect(issue).toHaveCount(0);
  await interactionOnly.press('Space');
  await expect(issue).toBeVisible();
});
