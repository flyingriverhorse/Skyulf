import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { mockBackend } from './fixtures/mockApi';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/drift-mixed-report.json', import.meta.url), 'utf8'));

test('keeps real producer PSI aligned across table, sparse history and CSV', async ({ page }) => {
  // The backend integration test pins this fixture to Core calculation and persisted API history.
  await mockBackend(page);
  const latest = fixture.history[0]!;
  const psi = latest.summary.category.psi;
  await page.route('**/api/monitoring/jobs', route => route.fulfill({ json: [{
    job_id: 'job-1', dataset_name: 'Mixed reference', filename: 'reference.csv',
  }] }));
  // Two older synthetic checks exercise the missing-column gap independently of the real newest check.
  await page.route('**/api/monitoring/drift/history/job-1', route => route.fulfill({ json: [
    latest,
    { ...latest, id: 2, summary: { amount: latest.summary.amount } },
    { ...latest, id: 3 },
  ] }));
  await page.route('**/api/monitoring/drift/calculate', route => route.fulfill({ json: fixture.report }));
  await page.goto('/drift');
  await page.getByRole('button', { name: /Select reference model/ }).click();
  await page.getByRole('option', { name: /Mixed reference/ }).click();
  await page.getByLabel('Upload current data file (CSV or Parquet)').setInputFiles({
    name: 'current.csv', mimeType: 'text/csv', buffer: Buffer.from('amount,category\n1,b\n'),
  });
  await page.getByRole('button', { name: 'Run Analysis', exact: true }).click();
  const category = page.getByRole('row').filter({ has: page.getByRole('cell', { name: 'category', exact: true }) });
  await expect(category.getByRole('cell', { name: psi.toFixed(4), exact: true })).toBeVisible();
  const sparkline = category.getByRole('img', { name: /PSI history/ });
  await expect(sparkline).toHaveAccessibleName(`PSI history: 1 missing checks; latest PSI ${psi}; threshold 0.2`);
  await expect(sparkline.locator('circle').last()).toHaveAttribute('fill', '#ef4444');

  await page.getByTitle('Drift thresholds', { exact: true }).click();
  await page.getByRole('spinbutton', { name: 'PSI', exact: true }).fill('0.5');
  await expect(category.getByText('Stable', { exact: true })).toBeVisible();
  await expect(sparkline.locator('circle').last()).toHaveAttribute('fill', '#22c55e');
  await expect(sparkline).toHaveAccessibleName(new RegExp('threshold 0.5$'));

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export CSV' }).click();
  const stream = await (await downloadPromise).createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const csv = Buffer.concat(chunks).toString('utf8');
  expect(csv).toContain(`"category","Stable","","${psi.toFixed(6)}"`);
  expect(csv).toContain('"amount","Stable","0.000000","0.000000"');
});
