import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('clustering exports the displayed summary as a PNG', async ({ page }) => {
  // Download must work for HTML summary cards without promising a nonexistent scatter plot.
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '310', source_id: 'clusters', name: 'Cluster summary', type: 'file', format: 'csv',
    rows: 4, columns: 2, created_at: '2026-09-21T00:00:00Z',
  }] } }));
  await page.route('**/api/eda/310/history', route => route.fulfill({ json: [] }));
  await page.route('**/api/eda/310/latest', route => route.fulfill({ json: {
    id: 310, status: 'COMPLETED', profile_data: {
      row_count: 4, column_count: 2, duplicate_rows: 0, missing_cells_percentage: 0,
      memory_usage_mb: 0.001, alerts: [], columns: {},
      clustering: { method: 'KMeans', n_clusters: 1, inertia: null,
        points: [{ x: 1, y: 2, cluster: 0 }],
        clusters: [{ cluster_id: 0, size: 4, percentage: 100, center: { amount: null, other: 3 } }],
      },
    },
  } }));
  await page.goto('/eda?dataset_id=310');
  await page.getByRole('button', { name: 'PCA & Clusters', exact: true }).click();
  await expect(page.getByText(/This summary shows the clustering method/)).toBeVisible();
  await expect(page.getByText('N/A', { exact: true })).toHaveCount(2);
  const pending = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download Summary', exact: true }).click();
  const download = await pending;
  expect(download.suggestedFilename()).toBe('clustering-summary.png');
  const png = await readFile((await download.path())!);
  expect(png.subarray(0, 8).toString('hex')).toBe('89504e470d0a1a0a');
  expect(png.readUInt32BE(16)).toBeGreaterThan(100);
  expect(png.readUInt32BE(20)).toBeGreaterThan(100);
});
