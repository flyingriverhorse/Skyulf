import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('keeps displayed metric and aggregation consistent with browser requests', async ({ page }) => {
  // The browser transport is mocked; the matching HTTP/Core fixture tests the arithmetic separately.
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '6161', source_id: 'revenue', name: 'Revenue', type: 'file', format: 'csv',
    rows: 2, columns: 1, created_at: '2026-09-20T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/6161/history', route => route.fulfill({ json: [] }));
  await page.route('**/api/eda/6161/latest', route => route.fulfill({ json: {
    id: 6161, status: 'COMPLETED', created_at: '2026-09-20T10:00:00Z',
    profile_data: {
      row_count: 2, column_count: 1, duplicate_rows: 0,
      missing_cells_percentage: 0, memory_usage_mb: 0.01, alerts: [],
      columns: { revenue: { name: 'revenue', dtype: 'Int64', missing_count: 0, missing_percentage: 0 } },
    },
  } }));
  const requests: Array<[string | null, string]> = [];
  await page.route('**/api/eda/6161/decomposition', route => {
    const body = route.request().postDataJSON();
    requests.push([body.measure_col, body.measure_agg]);
    const value = body.measure_col === null || body.measure_agg === 'count' ? 2 : body.measure_agg === 'sum' ? 30 : 15;
    return route.fulfill({ json: [{ name: 'Total', value, ratio: 1 }] });
  });
  await page.goto('/eda?dataset_id=6161');
  await page.getByRole('button', { name: 'Decomposition', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Total 2 (100%)' })).toBeVisible();
  const measure = page.locator('select').filter({ has: page.locator('option[value="count"]') });
  await measure.selectOption('revenue');
  const aggregation = page.locator('select').filter({ has: page.locator('option[value="mean"]') });
  await expect(aggregation).toHaveValue('sum');
  await expect(page.getByRole('button', { name: 'Total 30 (100%)' })).toBeVisible();
  await aggregation.selectOption('mean');
  await expect(page.getByRole('button', { name: 'Total 15 (100%)' })).toBeVisible();
  await measure.selectOption('count');
  await expect(aggregation).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Total 2 (100%)' })).toBeVisible();
  // React StrictMode replays mount effects in the development server.
  const transitions = requests.filter((entry, index) => JSON.stringify(entry) !== JSON.stringify(requests[index - 1]));
  expect(transitions).toEqual([[null, 'count'], ['revenue', 'sum'], ['revenue', 'mean']]);
});
