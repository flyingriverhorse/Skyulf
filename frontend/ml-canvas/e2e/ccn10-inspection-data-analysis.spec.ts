import { readFile } from 'node:fs/promises';
import { expect, test, type Locator, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import type { DriftReport } from '../src/core/api/monitoring';

/** Exercise the real preview service, including its schema-to-profile conversion. */
async function mockDatasetPreview(page: Page) {
  await mockBackend(page);
  const limits: string[] = [];
  const dataset = { id: 'ccn-preview', source_id: 'ccn-preview', name: 'CCN preview data',
    type: 'file', format: 'csv', rows: 3, columns: 3, size_bytes: 128,
    created_at: '2026-09-10T10:00:00Z', source_metadata: { ingestion_status: { status: 'completed' } } };
  await page.route('**/data/api/sources', route => route.fulfill({ json: { sources: [dataset] } }));
  await page.route('**/data/api/sources/ccn-preview/sample?*', route => {
    limits.push(new URL(route.request().url()).searchParams.get('limit')!);
    return route.fulfill({ json: { data: [
      { age: 0, active: false, species: null },
      { age: 10, active: true, species: 'setosa' },
      { age: 20, active: false, species: 'virginica' },
    ] } });
  });
  await page.route('**/api/pipeline/datasets/ccn-preview/schema', route => route.fulfill({ json: {
    row_count: 3, column_count: 3,
    columns: {
      age: { name: 'age', dtype: 'float64', column_type: 'numeric', missing_count: 0,
        missing_ratio: 0, unique_count: 3, mean_value: 10, std_value: 10, min_value: 0, max_value: 20 },
      active: { name: 'active', dtype: 'bool', column_type: 'categorical', missing_count: 0, unique_count: 2 },
      species: { name: 'species', dtype: 'object', column_type: 'categorical', missing_count: 1, missing_ratio: 1 / 3, unique_count: 2 },
    },
  } }));
  return limits;
}

/** Return a report whose only drift decision is PSI, so changing PSI has an observable effect. */
async function mockDriftCalculation(page: Page) {
  await mockBackend(page);
  const requests: string[] = [];
  await page.route('**/api/monitoring/jobs', route => route.fulfill({ json: [{
    job_id: 'ccn-reference', dataset_name: 'CCN reference data', filename: 'reference.csv',
    model_type: 'random_forest_classifier', target_column: 'species', n_features: 2, n_rows: 100,
  }] }));
  await page.route('**/api/monitoring/drift/history/ccn-reference', route => route.fulfill({ json: [] }));
  const report: DriftReport = {
    reference_rows: 100, current_rows: 150, drifted_columns_count: 1,
    missing_columns: [], new_columns: [], severity: 'warning',
    column_drifts: {
      height: { column: 'height', drift_detected: false, suggestions: [],
        metrics: [{ metric: 'psi', value: 0.05, threshold: 0.2, has_drift: false }] },
      age: { column: 'age', drift_detected: true, suggestions: ['Inspect the age distribution.'],
        metrics: [{ metric: 'psi', value: 0.3, threshold: 0.2, has_drift: true }],
        distribution: { bins: [
          { bin_start: 0, bin_end: 10, reference_count: 60, current_count: 50 },
          { bin_start: 10, bin_end: 20, reference_count: 40, current_count: 100 },
        ] },
      },
    },
  };
  await page.route('**/api/monitoring/drift/calculate', route => {
    requests.push(route.request().postData()!);
    return route.fulfill({ json: report });
  });
  return requests;
}

/** Check actual histogram bars after animation, not only the chart container. */
async function expectHistogram(chart: Locator) {
  await chart.scrollIntoViewIfNeeded();
  const bars = chart.locator('.recharts-bar-rectangle path');
  await expect(bars).toHaveCount(4);
  let previous = '';
  await expect.poll(async () => {
    const geometry = await bars.evaluateAll(elements => elements.map(element => {
      const bounds = element.getBoundingClientRect();
      return { d: element.getAttribute('d'), width: bounds.width, height: bounds.height };
    }));
    const current = JSON.stringify(geometry);
    const stable = current === previous;
    previous = current;
    return stable && geometry.every(bar => bar.width > 1 && bar.height > 1 && /^M/.test(bar.d ?? '') && !/NaN|Infinity/.test(bar.d ?? ''));
  }).toBe(true);
}

for (const width of [1440, 1100]) {
  test(`dataset sample and statistics retain values and reset on reopen at ${width}px`, async ({ page }, testInfo) => {
    // The real modal must retain false/zero values and submit the requested sample size.
    await page.setViewportSize({ width, height: 1000 });
    const limits = await mockDatasetPreview(page);
    await page.goto('/data');
    await page.getByRole('textbox', { name: 'Search datasets' }).fill('CCN preview');
    const preview = page.getByRole('button', { name: 'Preview dataset', exact: true });
    await preview.click();
    const modal = page.getByRole('dialog');
    const firstRow = modal.getByRole('row').nth(1);
    await expect(firstRow.getByRole('cell')).toHaveText(['0', 'false', '']);
    await modal.getByRole('button', { name: 'Load More (+500 rows)', exact: true }).click();
    await expect.poll(() => limits).toEqual(['100', '600']);
    await expect(modal.getByText('Showing first 3 rows', { exact: true })).toBeVisible();
    await modal.getByRole('button', { name: 'Statistics', exact: true }).click();
    const ageRow = modal.getByRole('row').filter({ has: page.getByRole('cell', { name: 'age', exact: true }) });
    await expect(ageRow.getByRole('cell')).toHaveText(['age', 'float64', '0 (0%)', '3', '0.00', '20.00', '10.00', '10.00']);
    await expect.poll(() => page.evaluate(() => document.getAnimations().every(animation => animation.playState === 'finished'))).toBe(true);
    await page.screenshot({ path: testInfo.outputPath(`preview-statistics-${width}.png`) });
    await modal.getByText('Close', { exact: true }).click();
    await expect(modal).toBeHidden();
    await preview.click();
    await expect(modal.getByRole('row').nth(1).getByRole('cell')).toHaveText(['0', 'false', '']);
    await expect.poll(() => limits).toEqual(['100', '600', '100']);
  });

  test(`drift upload, chart, sort, thresholds and CSV stay connected at ${width}px`, async ({ page }, testInfo) => {
    // Client-side threshold changes must update filtering/export without another calculation request.
    await page.setViewportSize({ width, height: 1000 });
    const requests = await mockDriftCalculation(page);
    await page.goto('/drift');
    await page.getByRole('button', { name: /Select reference model/ }).click();
    await page.getByRole('option', { name: /CCN reference data/ }).click();
    await page.getByLabel('Upload current data file (CSV or Parquet)').setInputFiles({
      name: 'current.csv', mimeType: 'text/csv', buffer: Buffer.from('age,height\n25,170\n'),
    });
    await page.getByRole('button', { name: 'Run Analysis', exact: true }).click();
    await expect(page.getByText('50% of features', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(1);
    expect(requests[0]).toContain('ccn-reference');
    expect(requests[0]).toContain('filename="current.csv"');
    expect(requests[0]).toContain('name="threshold_psi"\r\n\r\n0.2');
    const table = page.getByRole('table');
    await table.getByRole('columnheader', { name: 'Column', exact: true }).click();
    await expect(table.locator('tbody tr > td:first-child')).toHaveText(['age', 'height']);
    const ageRow = table.getByRole('row').filter({ has: page.getByRole('cell', { name: 'age', exact: true }) });
    await ageRow.getByRole('button', { name: 'Details', exact: true }).click();
    await expect(page.getByText('Inspect the age distribution.', { exact: true })).toBeVisible();
    await expectHistogram(table.locator('.recharts-wrapper'));
    await table.evaluate(element => { element.parentElement!.scrollLeft = 0; });
    await page.screenshot({ path: testInfo.outputPath(`drift-distribution-${width}.png`), fullPage: true });
    await ageRow.getByRole('button', { name: 'Hide', exact: true }).click();
    await page.getByRole('button', { name: 'Show only drifted', exact: true }).click();
    await expect(table.locator('tbody tr > td:first-child')).toHaveText(['age']);
    await page.getByTitle('Drift thresholds', { exact: true }).click();
    await page.getByRole('spinbutton', { name: 'PSI', exact: true }).fill('0.4');
    await expect(page.getByText('0% of features', { exact: true })).toBeVisible();
    await expect(table.getByRole('cell', { name: 'age', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Showing drifted only', exact: true }).click();
    await expect(table.locator('tbody tr > td:first-child')).toHaveText(['age', 'height']);
    const downloaded = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export CSV', exact: true }).click();
    const download = await downloaded;
    const csv = await readFile((await download.path())!, 'utf8');
    expect(download.suggestedFilename()).toContain('drift_report_CCN reference data_');
    expect(csv).toContain('"age","Stable"');
    expect(csv).toContain('"height","Stable"');
    await page.getByRole('button', { name: 'Reset defaults', exact: true }).click();
    await expect(page.getByText('50% of features', { exact: true })).toBeVisible();
    expect(requests).toHaveLength(1);
  });
}
