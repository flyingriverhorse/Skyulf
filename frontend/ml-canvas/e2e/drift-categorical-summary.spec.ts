import { expect, test, type Page } from '@playwright/test';
import type { DriftReport } from '../src/core/api/monitoring';
import { mockBackend } from './fixtures/mockApi';

/** Preserve the calculate endpoint's categorical metric name through the real page. */
async function mockCategoricalDrift(page: Page) {
  await mockBackend(page);
  await page.route('**/api/monitoring/jobs', route => route.fulfill({ json: [{
    job_id: 'categorical-reference', dataset_name: 'Categorical reference', filename: 'reference.csv',
    model_type: 'random_forest_classifier', target_column: 'target', n_features: 2, n_rows: 100,
  }] }));
  await page.route('**/api/monitoring/drift/history/categorical-reference', route => route.fulfill({ json: [] }));
  const report: DriftReport = {
    reference_rows: 100, current_rows: 100, drifted_columns_count: 1,
    missing_columns: [], new_columns: [], severity: 'critical',
    column_drifts: {
      numeric: { column: 'numeric', drift_detected: false, suggestions: [],
        metrics: [{ metric: 'psi', value: 0.01, threshold: 0.2, has_drift: false }] },
      category: { column: 'category', drift_detected: true,
        suggestions: ['Inspect the category distribution.'],
        metrics: [{ metric: 'psi_categorical', value: 5, threshold: 0.2, has_drift: true }] },
    },
  };
  await page.route('**/api/monitoring/drift/calculate', route => route.fulfill({ json: report }));
}

for (const width of [1440, 390]) {
  test(`categorical drift drives the summary after an upload at ${width}px`, async ({ page }, testInfo) => {
    /** The upload response, threshold evaluation, table and cards must agree on categorical PSI. */
    await page.setViewportSize({ width, height: 1000 });
    await mockCategoricalDrift(page);
    await page.goto('/drift');
    await page.getByRole('button', { name: /Select reference model/ }).click();
    await page.getByRole('option', { name: /Categorical reference/ }).click();
    await page.getByLabel('Upload current data file (CSV or Parquet)').setInputFiles({
      name: 'current.csv', mimeType: 'text/csv', buffer: Buffer.from('numeric,category\n1,b\n'),
    });
    await page.getByRole('button', { name: 'Run Analysis', exact: true }).click();

    const average = page.getByText('Avg PSI', { exact: true }).locator('..');
    const mostDrifted = page.getByText('Most Drifted', { exact: true }).locator('..');
    await expect(average.getByText('2.5050', { exact: true })).toBeVisible();
    await expect(average.getByText('Significant drift', { exact: true })).toBeVisible();
    await expect(mostDrifted.getByText('category', { exact: true })).toBeVisible();
    await expect(mostDrifted.getByText('PSI: 5.0000', { exact: true })).toBeVisible();
    await expect(page.getByText('50% of features', { exact: true })).toBeVisible();
    const categoryRow = page.getByRole('row').filter({ has: page.getByRole('cell', { name: 'category', exact: true }) });
    await expect(categoryRow.getByRole('cell', { name: '5.0000', exact: true })).toBeVisible();

    const cards = average.locator('..');
    await cards.scrollIntoViewIfNeeded();
    await cards.screenshot({ path: testInfo.outputPath(`categorical-drift-cards-${width}.png`) });
    const bounds = await cards.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(width);
  });
}
