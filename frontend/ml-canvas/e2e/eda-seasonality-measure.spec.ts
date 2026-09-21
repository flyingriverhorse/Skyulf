import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('shows actual seasonality measures in chart captions and tooltips, including legacy reports', async ({ page }) => {
  /** The legacy count wire key must never label a known mean as a row count. */
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '302', source_id: 'seasonality', name: 'Seasonality', type: 'file', format: 'parquet',
    rows: 2, columns: 2, created_at: '2026-09-21T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/302/history', route => route.fulfill({ json: [] }));
  let measure: Record<string, string | null> = { aggregation: 'mean', metric: 'sales' };
  await page.route('**/api/eda/302/latest', route => route.fulfill({ json: {
    id: 302, status: 'COMPLETED', created_at: '2026-09-21T10:00:00Z', profile_data: {
      row_count: 2, column_count: 2, columns: {}, alerts: [], duplicate_rows: 0,
      missing_cells_percentage: 0, memory_usage_mb: 1,
      timeseries: { date_col: 'date', trend: [], seasonality: {
        day_of_week: [{ day: 'Mon', count: 20 }], month_of_year: [{ month: 'Jan', count: 20 }],
        ...measure,
      } },
    },
  } }));
  for (const [metadata, label] of [
    [{ aggregation: 'mean', metric: 'sales' }, 'Mean of sales'],
    [{ aggregation: 'count', metric: null }, 'Row count'],
    [{}, 'Recorded value (measure unavailable)'],
  ] as const) {
    measure = metadata;
    await page.goto('/eda?dataset_id=302');
    await page.getByRole('button', { name: 'Time Series', exact: true }).click();
    await expect(page.getByText(label, { exact: true })).toHaveCount(2);
    for (const id of ['day-seasonality-chart', 'month-seasonality-chart']) {
      const chart = page.locator(`#${id}`);
      await chart.locator('.recharts-bar-rectangle').hover();
      await expect(chart.locator('.recharts-tooltip-item-name')).toHaveText(label);
      await expect(chart.locator('.recharts-tooltip-item-value')).toHaveText('20');
    }
  }
});
