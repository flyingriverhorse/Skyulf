import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const width of [1440, 390]) {
  test.describe(`EDA empty states at ${width}px`, () => {
    test.use({ viewport: { width, height: 1000 } });

    test('distinguishes zero outliers, missing results and recorded failure', async ({ page }) => {
      /** Switching saved reports must not turn missing analysis into a successful zero result. */
      await mockBackend(page);
      await page.route('**/api/eda/302/decomposition', route => route.fulfill({ json: [] }));
      await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
        id: '302', source_id: 'empty-states', name: 'Empty states', type: 'file', format: 'parquet',
        rows: 100, columns: 1, created_at: '2026-09-21T10:00:00Z',
      }] } }));
      await page.route('**/api/eda/302/history', route => route.fulfill({ json: [{
        id: 301, status: 'COMPLETED', target_col: 'measurement', created_at: '2026-09-20T10:00:00Z',
      }] }));
      await page.route('**/api/eda/reports/301', route => route.fulfill({ json: {
        id: 301, status: 'COMPLETED', created_at: '2026-09-20T10:00:00Z', profile_data: {
          row_count: 100, column_count: 1, columns: {}, alerts: [], duplicate_rows: 0,
          missing_cells_percentage: 0, memory_usage_mb: 1, target_col: 'measurement',
        },
      } }));
      let state = 'zero';
      await page.route('**/api/eda/302/latest', route => route.fulfill({ json: {
        id: 302, status: state === 'failed' ? 'FAILED' : 'COMPLETED',
        created_at: '2026-09-21T10:00:00Z', error_message: state === 'failed' ? 'Recorded analysis failure' : null,
        profile_data: {
          row_count: 100, column_count: 1, duplicate_rows: 0, missing_cells_percentage: 0, memory_usage_mb: 1,
          alerts: [], columns: { measurement: { name: 'measurement', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 } },
          outliers: state === 'zero' ? {
            method: 'IsolationForest', total_outliers: 0, outlier_percentage: 0,
            analyzed_rows: 100, total_rows: 100, top_outliers: [],
          } : null,
        },
      } }));
      await page.goto('/eda?dataset_id=302');
      if (width === 390) await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
      for (const [tab, message] of [
        ['Dashboard', 'Column Types'],
        ['PII Review', 'No PII findings recorded'],
        ['Smart Insights', 'No recommendations were recorded for this report.'],
        ['Variables', 'Showing 1 of 1 variables'],
        ['Bivariate', 'Sample data is required for bivariate analysis.'],
      ]) {
        await page.getByRole('button', { name: tab, exact: true }).click();
        await expect(page.getByText(message, { exact: true })).toBeVisible();
      }
      await page.getByRole('button', { name: 'Decomposition', exact: true }).click();
      await expect(page.getByRole('button', { name: 'Reset Tree', exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Outliers', exact: true }).click();
      await expect(page.getByText('No outliers detected', { exact: true })).toBeVisible();
      await expect(page.getByText(/Analyzed all 100 rows/)).toBeVisible();

      await page.getByRole('button', { name: 'measurement', exact: true }).click();
      await expect(page.getByText('Analysis results unavailable', { exact: true })).toBeVisible();
      await expect(page.getByText(/No reason was recorded/)).toBeVisible();
      await expect(page.getByText('No outliers detected', { exact: true })).toHaveCount(0);

      state = 'failed';
      await page.reload();
      await expect(page.getByText('Analysis Failed', { exact: true })).toBeVisible();
      await expect(page.getByText('Recorded analysis failure', { exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Retry', exact: true })).toBeVisible();
    });
  });
}
