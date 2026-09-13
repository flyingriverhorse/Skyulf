import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Load report metadata through the same endpoints used by a saved EDA report. */
async function mockProfile(page: Page, columns: Record<string, unknown>, extra: Record<string, unknown> = {}) {
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '193', source_id: 'profile-roundtrip', name: 'Profile roundtrip', type: 'file', format: 'parquet',
    rows: 4, columns: Object.keys(columns).length, created_at: '2026-09-13T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/193/history', route => route.fulfill({ json: [] }));
  await page.route('**/api/eda/193/latest', route => route.fulfill({ json: {
    id: 193, status: 'COMPLETED', created_at: '2026-09-13T10:00:00Z',
    profile_data: {
      row_count: 4, column_count: Object.keys(columns).length, duplicate_rows: 0,
      missing_cells_percentage: 0, memory_usage_mb: 0.01, alerts: [], columns, ...extra,
    },
  } }));
}

/** Use normal page navigation and leave enough chart space on mobile. */
async function openAnalysis(page: Page, name: 'Correlations' | 'Decomposition', mobile: boolean) {
  await page.goto('/eda?dataset_id=193');
  if (mobile) await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
  await page.getByRole('button', { name, exact: true }).click();
}

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(`EDA profile roundtrip on ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('explains backend correlation omissions without promising absent table data', async ({ page }) => {
      /** A capped backend response must expose omissions even when the rendered matrix has only 20 columns. */
      const allColumns = Array.from({ length: 25 }, (_, index) => `feature_${index}`);
      const matrixColumns = allColumns.slice(0, 20);
      await mockProfile(page, Object.fromEntries(allColumns.map(name => [name, {
        name, dtype: 'Numeric', missing_count: 0, missing_percentage: 0,
      }])), {
        correlations: {
          columns: matrixColumns,
          values: matrixColumns.map((_, row) => matrixColumns.map((__, column) => row === column ? 1 : 0.5)),
          total_columns: 25, omitted_columns: allColumns.slice(20),
        },
      });
      await openAnalysis(page, 'Correlations', viewport.name === 'mobile');
      const warning = page.getByText(/Correlation analysis used the first 20 of 25 numeric columns/);
      await expect(warning).toBeVisible();
      await expect(warning).toContainText('5 omitted: feature_20, feature_21, feature_22, feature_23, feature_24');
      await expect(warning).toContainText('These omitted columns are not included in the data table.');
      await expect(page.getByText(/Use the data table below for the full matrix/)).toHaveCount(0);
      await page.getByRole('button', { name: 'View data table', exact: true }).click();
      const table = page.getByRole('region', { name: 'Full correlation matrix data table', exact: true });
      await expect(table).toBeVisible();
      await expect(table.getByRole('columnheader')).toHaveText(['Variable', ...matrixColumns]);
      await expect(table.getByRole('row')).toHaveCount(21);
      await expect(table).not.toContainText('feature_24');
    });

    test('retains timezone and fractional seconds when drilling into a temporal bucket', async ({ page }) => {
      /** The selected bucket's wire value must survive page controls without browser date normalization. */
      const timestamp = '2026-01-01 11:30:00.123456+02:00';
      await mockProfile(page, {
        recorded_at: { name: 'recorded_at', dtype: 'DateTime', missing_count: 0, missing_percentage: 0 },
        detail: {
          name: 'detail', dtype: 'Categorical', missing_count: 0, missing_percentage: 0,
          categorical_stats: { unique_count: 2, top_k: [], rare_labels_count: 0 },
        },
      });
      const requests: Array<{ column: string; operator: string; value: string }> = [];
      await page.route('**/api/eda/193/decomposition', route => {
        const body = route.request().postDataJSON();
        if (!body.split_col) return route.fulfill({ json: [{ name: 'Total', value: 4, ratio: 1 }] });
        if (body.split_col === 'recorded_at') return route.fulfill({ json: [
          { name: timestamp, filter_value: timestamp, value: 2, ratio: 0.5 },
          { name: '2026-01-02 11:30:00+02:00', filter_value: '2026-01-02 11:30:00+02:00', value: 2, ratio: 0.5 },
        ] });
        requests.push(...body.filters);
        if (body.split_col === 'detail' && body.filters[0]?.value === timestamp) {
          return route.fulfill({ json: [{ name: 'Matching detail', filter_value: 'Matching detail', value: 2, ratio: 1 }] });
        }
        return route.fulfill({ status: 400, json: { detail: 'Unexpected temporal filter' } });
      });
      await openAnalysis(page, 'Decomposition', viewport.name === 'mobile');
      await page.getByRole('button', { name: 'Total 4 (100%)', exact: true }).click();
      await page.getByTitle('Split further').click();
      await page.getByRole('button', { name: 'recorded_at', exact: true }).click();
      const bucket = page.getByRole('button', { name: `${timestamp} 2 (50%)`, exact: true });
      await bucket.focus();
      await bucket.press('Enter');
      await expect(bucket).toHaveAttribute('aria-pressed', 'true');
      await page.getByTitle('Split further').click();
      await page.getByRole('button', { name: 'detail', exact: true }).click();
      await expect(page.getByText('Matching detail', { exact: true })).toBeVisible();
      expect(requests).toEqual([{ column: 'recorded_at', operator: '==', value: timestamp }]);
    });
  });
}
