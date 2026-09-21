import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('excludes missing predictions from statistics and histogram while retaining result rows', async ({ page }) => {
  // The real controller and rendered histogram must share the observed numeric population.
  await mockBackend(page);
  await page.route('**/api/deployment/active', route => route.fulfill({ json: {
    id: 1, job_id: 'numeric-stats', model_type: 'regressor', artifact_uri: 'model.joblib',
    is_active: true, created_at: '2026-09-21T00:00:00Z', input_schema: [{ name: 'a', type: 'float64' }],
  } }));
  await page.route('**/api/pipeline/jobs/numeric-stats', route => route.fulfill({ json: {
    job_id: 'numeric-stats', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
    status: 'completed', start_time: null, end_time: null, error: null, result: null,
    created_at: '2026-09-21T00:00:00Z',
  } }));
  let predictions: unknown[] = [1, null, 3];
  await page.route('**/api/deployment/predict', route => route.fulfill({ json: { predictions, model_version: 'v1' } }));
  await page.goto('/canvas');
  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  const editor = page.locator('textarea[aria-labelledby="inference-input-heading"]');
  await expect(editor).toHaveValue(/"a"/);
  await editor.fill('[{"a":1},{"a":2},{"a":3}]');
  await page.getByRole('button', { name: 'Run Prediction', exact: true }).click();
  await expect(page.getByText('n = 2', { exact: true })).toBeVisible();
  await expect(page.getByText('mean 2.0000', { exact: true })).toBeVisible();
  await expect(page.getByText('min 1.0000', { exact: true })).toBeVisible();
  const histogram = page.getByTitle('Distribution of predictions', { exact: true });
  await expect(histogram.getByText('1.00', { exact: true })).toBeVisible();
  await expect(histogram.locator('rect:not([height="0"])')).toHaveCount(2);
  await page.getByRole('button', { name: 'Table', exact: true }).click();
  await expect(page.locator('table tbody tr')).toHaveCount(3);

  predictions = [null, '', false];
  await page.getByRole('button', { name: 'Run Prediction', exact: true }).click();
  await expect(page.getByTestId('run-provenance')).toContainText('Run #2');
  await expect(page.getByText(/^mean /)).toHaveCount(0);
  await expect(histogram).toHaveCount(0);
  await expect(page.locator('table tbody tr')).toHaveCount(3);
});
