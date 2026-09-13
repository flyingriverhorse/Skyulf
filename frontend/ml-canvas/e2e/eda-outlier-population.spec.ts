import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(`outlier population on ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('shows sample scope and keeps legacy denominator unknown on report reload', async ({ page }) => {
      /** Normal EDA navigation must present the stored sample population and clear it for older reports. */
      await mockBackend(page);
      await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
        id: '302', source_id: 'outlier-population', name: 'Outlier population', type: 'file', format: 'parquet',
        rows: 200000, columns: 1, created_at: '2026-09-13T10:00:00Z',
      }] } }));
      await page.route('**/api/eda/302/history', route => route.fulfill({ json: [] }));
      let population: Record<string, number> = { analyzed_rows: 50000, total_rows: 200000 };
      await page.route('**/api/eda/302/latest', route => route.fulfill({ json: {
        id: 302, status: 'COMPLETED', created_at: '2026-09-13T10:00:00Z',
        profile_data: {
          row_count: 200000, column_count: 1, duplicate_rows: 0,
          missing_cells_percentage: 0, memory_usage_mb: 1.6, alerts: [],
          columns: { measurement: { name: 'measurement', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 } },
          outliers: {
            method: 'IsolationForest', total_outliers: 2497, outlier_percentage: 4.994,
            top_outliers: [{ index: 123456, values: { measurement: 12 }, score: -0.1 }],
            ...population,
          },
        },
      } }));
      await page.goto('/eda?dataset_id=302');
      if (viewport.name === 'mobile') await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
      await page.getByRole('button', { name: 'Outliers', exact: true }).click();
      await expect(page.getByText(/Analyzed 50,000 sampled rows out of 200,000 rows/)).toBeVisible();
      await expect(page.getByText('Outliers in sample', { exact: true })).toBeVisible();
      await expect(page.getByText('Percentage of sampled rows', { exact: true })).toBeVisible();
      await expect(page.getByText('2497', { exact: true })).toBeVisible();
      await expect(page.getByText('4.99%', { exact: true })).toBeVisible();
      await expect(page.getByRole('cell', { name: '123456', exact: true })).toBeVisible();

      population = {};
      await page.reload();
      if (viewport.name === 'mobile') await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
      await page.getByRole('button', { name: 'Outliers', exact: true }).click();
      await expect(page.getByText(/Analyzed row count is unavailable for this saved report/)).toBeVisible();
      await expect(page.getByText('Reported outliers', { exact: true })).toBeVisible();
      await expect(page.getByText(/Analyzed 50,000 sampled rows/)).toHaveCount(0);
    });
  });
}
