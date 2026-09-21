import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/eda-null-producer.json', import.meta.url), 'utf8'));

for (const width of [1440, 390]) {
  test(`real null statistics survive saved report navigation at ${width}px`, async ({ page }) => {
    // Python tests pin these projections to actual analyzer output and JSON serialization.
    await page.setViewportSize({ width, height: 1000 });
    await mockBackend(page);
    await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
      id: '309', source_id: 'null-statistics', name: 'Null statistics', type: 'file', format: 'csv',
      rows: 1, columns: 1, created_at: '2026-09-21T00:00:00Z',
    }] } }));
    await page.route('**/api/eda/309/history', route => route.fulfill({ json: [] }));
    let outlierReport = false;
    await page.route('**/api/eda/309/latest', route => route.fulfill({ json: {
      id: 309, status: 'COMPLETED', profile_data: {
        row_count: outlierReport ? 100 : 1, column_count: 1, duplicate_rows: 0,
        missing_cells_percentage: 0, memory_usage_mb: 0.001, alerts: [],
        columns: outlierReport ? { tiny: { name: 'tiny', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 } }
          : { value: fixture.singleton_column },
        outliers: outlierReport ? fixture.outliers : null,
      },
    } }));
    await page.goto('/eda?dataset_id=309');
    if (width === 390) await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
    await page.getByRole('button', { name: 'Variables', exact: true }).click();
    await page.getByRole('button', { name: 'Expand All', exact: true }).click();
    await expect(page.getByText('Variance', { exact: true }).locator('..')).toHaveText('Variance—');
    await expect(page.getByText('Std Dev', { exact: true }).locator('..')).toHaveText('Std Dev—');
    outlierReport = true;
    await page.reload();
    if (width === 390) await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
    await page.getByRole('button', { name: 'Outliers', exact: true }).click();
    await expect(page.getByText(/Diff: —/)).toBeVisible();
    await expect(page.getByRole('cell', { name: '99', exact: true })).toBeVisible();
    await expect(page.getByText(/Something went wrong/)).toHaveCount(0);
  });
}
