import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('shows native JSON explanations with only the coordinates the browser provides', async ({ page }) => {
  /** Syntax errors must block prediction without manufacturing a location. */
  await mockBackend(page);
  await page.route('**/api/deployment/active', route => route.fulfill({ json: {
    id: 1, job_id: 'json-job', model_type: 'regressor', artifact_uri: 'model.joblib',
    is_active: true, created_at: '2026-09-21T00:00:00Z', input_schema: [{ name: 'a', type: 'float64' }],
  } }));
  await page.route('**/api/pipeline/jobs/json-job', route => route.fulfill({ json: {
    job_id: 'json-job', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
    status: 'completed', target_column: 'target', dropped_columns: [], created_at: '2026-09-21T00:00:00Z',
  } }));
  await page.goto('/canvas');
  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  const editor = page.locator('textarea[aria-labelledby="inference-input-heading"]');
  await expect(editor).toHaveValue(/"a"/);
  await editor.fill('{"a":1,}');
  await expect(page.locator('#inference-input-status')).toContainText('(line 1, col 8)');
  await expect(page.getByRole('button', { name: 'Run Prediction', exact: true })).toBeDisabled();

  const nativeMessage = await page.evaluate(() => {
    try { JSON.parse('undefined'); } catch (error) { return (error as Error).message; }
    throw new Error('Invalid JSON was accepted');
  });
  await editor.fill('undefined');
  await expect(page.locator('#inference-input-status')).toHaveText(nativeMessage);
  await expect(page.locator('#inference-input-status')).not.toContainText(/line \d|col \d/);
  await expect(page.getByRole('button', { name: 'Run Prediction', exact: true })).toBeDisabled();
});
