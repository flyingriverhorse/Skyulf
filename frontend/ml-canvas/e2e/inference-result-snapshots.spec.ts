import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
  test(`exports the original prediction inputs after editor changes at ${viewport.width}px`, async ({ page }) => {
    // Real downloads and table cells must retain the same request provenance.
    await page.setViewportSize(viewport);
    await mockBackend(page);
    await page.route('**/api/deployment/active', route => route.fulfill({ json: {
      id: 1, job_id: 'job-snapshot', model_type: 'regressor', artifact_uri: 'model.joblib',
      is_active: true, created_at: '2026-09-12T00:00:00Z', input_schema: [{ name: 'a', type: 'float64' }],
    } }));
    await page.route('**/api/pipeline/jobs/job-snapshot', route => route.fulfill({ json: {
      job_id: 'job-snapshot', pipeline_id: 'pipeline', node_id: 'model', job_type: 'training',
      status: 'completed', start_time: null, end_time: null, error: null, result: null,
      created_at: '2026-09-12T00:00:00Z',
    } }));
    await page.route('**/api/deployment/predict', route => {
      const body = route.request().postDataJSON() as { data: { a: number }[] };
      return route.fulfill({ json: { predictions: body.data.map(row => row.a * 10), model_version: 'v1' } });
    });
    await page.goto('/canvas');
    await page.getByRole('tab', { name: 'Inference', exact: true }).click();
    const editor = page.locator('textarea[aria-labelledby="inference-input-heading"]');
    await expect(editor).toHaveValue(/"a"/);
    await editor.fill('[{"a":1},{"a":2}]');
    await page.getByRole('button', { name: 'Run Prediction', exact: true }).click();
    await expect(page.getByTestId('run-provenance')).toContainText('2 rows');
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await editor.fill('[{"a":9},{"a":8},{"a":7}]');
    const rows = page.locator('table tbody tr');
    await expect(rows).toHaveCount(2);
    await expect(rows.nth(0).locator('td')).toHaveText(['1', '1', '10']);
    await expect(rows.nth(1).locator('td')).toHaveText(['2', '2', '20']);
    const downloadPromise = page.waitForEvent('download');
    await page.getByTitle('Download inputs + predictions as CSV', { exact: true }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/^predictions_job-snapshot_run-.*\.csv$/);
    expect(await readFile((await download.path())!, 'utf8')).toBe('a,prediction\n1,10\n2,20');
    await editor.fill('{invalid JSON');
    await expect(rows).toHaveCount(2);
    await expect(rows.nth(0).locator('td')).toHaveText(['1', '1', '10']);
  });
}
