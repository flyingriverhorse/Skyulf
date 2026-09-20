import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('CSV upload, prediction payload and downloaded reimport retain quoted cells', async ({ page }) => {
  // Browser FileReader/downloads are real; only backend transport is controlled.
  await mockBackend(page);
  await page.route('**/api/deployment/active', route => route.fulfill({ json: {
    id: 1, job_id: 'csv-job', model_type: 'regressor', artifact_uri: 'model.joblib',
    is_active: true, created_at: '2026-09-20T00:00:00Z',
    input_schema: [{ name: 'amount', type: 'float64' }, { name: 'note,detail', type: 'string' }],
  } }));
  await page.route('**/api/pipeline/jobs/csv-job', route => route.fulfill({ json: {
    job_id: 'csv-job', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
    status: 'completed', target_column: 'target', dropped_columns: [],
    created_at: '2026-09-20T00:00:00Z',
  } }));
  let submitted: unknown;
  await page.route('**/api/deployment/predict', route => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: { predictions: [10, 20], model_version: 'v1' } });
  });
  await page.goto('/canvas');
  await page.getByRole('tab', { name: 'Inference', exact: true }).click();
  const editor = page.locator('textarea[aria-labelledby="inference-input-heading"]');
  await expect(editor).toHaveValue(/amount/);
  const csv = 'amount,"note,detail",target\r\n1.5,"first,line\r\nsaid ""hi""",yes\r\n2,"",no';
  await page.locator('input[type="file"]').setInputFiles({ name: 'quoted.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) });
  const expected = [{ amount: 1.5, 'note,detail': 'first,line\r\nsaid "hi"' }, { amount: 2, 'note,detail': '' }];
  await expect.poll(async () => JSON.parse(await editor.inputValue())).toEqual(expected);
  await page.getByRole('button', { name: 'Run Prediction', exact: true }).click();
  await expect(page.getByTestId('run-provenance')).toContainText('2 rows');
  expect(submitted).toMatchObject({ data: expected });
  const pending = page.waitForEvent('download');
  await page.getByTitle('Download inputs + predictions as CSV', { exact: true }).click();
  const download = await pending;
  const exported = await readFile((await download.path())!, 'utf8');
  await page.locator('input[type="file"]').setInputFiles({ name: 'roundtrip.csv', mimeType: 'text/csv', buffer: Buffer.from(exported) });
  const exportedRows = expected.map((row, index) => ({ ...row, prediction: [10, 20][index] }));
  await expect.poll(async () => JSON.parse(await editor.inputValue())).toEqual(exportedRows);
  await page.locator('input[type="file"]').setInputFiles({ name: 'bad.csv', mimeType: 'text/csv', buffer: Buffer.from('amount,note\n1,2,3') });
  await expect(page.getByText(/CSV row 2: expected 2 cells, received 3/)).toBeVisible();
  expect(JSON.parse(await editor.inputValue())).toEqual(exportedRows);
});
