import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('loads inference on first visit and preserves unsaved form state across views', async ({ page }) => {
  // Canvas startup must avoid inference code; later navigation must retain its mounted form.
  await mockBackend(page);
  await page.route('**/api/deployment/active', route => route.fulfill({ json: {
    id: 1, job_id: '', model_type: 'random_forest_classifier', artifact_uri: 'model.joblib',
    is_active: true, created_at: '2026-09-09T12:00:00Z',
    input_schema: [{ name: 'required_feature', type: 'float64' }],
  } }));
  let inferenceRequests = 0;
  page.on('request', request => {
    if (new URL(request.url()).pathname.endsWith('/InferencePage.tsx')) inferenceRequests++;
  });
  await page.goto('/canvas');
  await expect(page.locator('.react-flow')).toBeVisible();
  expect(inferenceRequests).toBe(0);

  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  const input = page.locator('textarea[aria-labelledby="inference-input-heading"]');
  await expect(input).toBeVisible();
  await expect(input).toHaveValue(/required_feature/);
  await input.fill('[{"other_feature": 42}]');
  const acknowledge = page.getByTestId('acknowledge-missing-fields');
  await acknowledge.check();
  expect(inferenceRequests).toBe(1);

  await page.getByRole('tab', { name: 'Canvas', exact: true }).click();
  await expect(input).toBeHidden();
  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  await expect(input).toHaveValue('[{"other_feature": 42}]');
  await expect(acknowledge).toBeChecked();
  expect(inferenceRequests).toBe(1);
});
