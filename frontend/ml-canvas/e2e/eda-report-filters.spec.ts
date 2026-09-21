import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

test('restores report filter chips and keeps decomposition cache and reset in the selected population', async ({ page }) => {
  // Real browser controls exercise report loading and HTTP payloads; server responses are controlled.
  await mockBackend(page);
  const eu = { column: 'region', operator: '==', value: 'EU' };
  const us = { ...eu, value: 'US' };
  const profile = {
    row_count: 2, column_count: 1, duplicate_rows: 0,
    missing_cells_percentage: 0, memory_usage_mb: 0.01, alerts: [],
    columns: { region: { name: 'region', dtype: 'Categorical', missing_count: 0, missing_percentage: 0 } },
  };
  const latest = { id: 7601, status: 'COMPLETED', created_at: '2026-09-21T10:00:00Z', profile_data: profile, config: { filters: [eu] } };
  const historical = { ...latest, id: 7600, config: { filters: [us] } };
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '7601', source_id: 'regions', name: 'Regions', type: 'file', format: 'csv',
    rows: 4, columns: 1, created_at: latest.created_at,
  }] } }));
  await page.route('**/api/eda/7601/latest', route => route.fulfill({ json: latest }));
  await page.route('**/api/eda/7601/history', route => route.fulfill({ json: [latest, historical] }));
  await page.route('**/api/eda/reports/7600', route => route.fulfill({ json: historical }));
  await page.route('**/api/eda/reports/7601', route => route.fulfill({ json: latest }));
  const requests: string[] = [];
  let refreshed = false;
  await page.route('**/api/eda/7601/decomposition', route => {
    const { filters } = route.request().postDataJSON();
    const region = filters[0]?.value ?? 'unfiltered';
    requests.push(region);
    return route.fulfill({ json: [{ name: `${region} ${refreshed ? 'fresh' : 'total'}`, value: 2, ratio: 1 }] });
  });
  await page.goto('/eda?dataset_id=7601');
  await expect(page.getByRole('button', { name: 'Draft Filters (1)' })).toBeVisible();
  await expect(page.getByText('EU', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Remove filter 1' }).click();
  await page.getByRole('button', { name: 'Reset filters', exact: true }).click();
  await expect(page.getByText('EU', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Decomposition', exact: true }).click();
  await expect(page.getByText('EU total', { exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'History', exact: true }).click();
  await page.getByRole('button', { name: /Analysis #7600/ }).click();
  await page.getByRole('button', { name: 'Load this Report' }).click();
  await expect(page.getByText('US', { exact: true })).toBeVisible();
  await expect(page.getByText('US total', { exact: true })).toBeVisible();

  refreshed = true;
  await page.getByRole('button', { name: 'Reset Tree' }).click();
  await expect(page.getByText('US fresh', { exact: true })).toBeVisible();
  const requestsAfterReset = requests.length;
  await page.getByRole('button', { name: 'Dashboard', exact: true }).click();
  await page.getByRole('button', { name: 'Decomposition', exact: true }).click();
  await expect(page.getByText('US fresh', { exact: true })).toBeVisible();
  expect(requests).toHaveLength(requestsAfterReset);

  await page.getByRole('button', { name: 'History', exact: true }).click();
  await page.getByRole('button', { name: /Analysis #7601/ }).click();
  await page.getByRole('button', { name: 'Load this Report' }).click();
  await expect(page.getByText('EU total', { exact: true })).toBeVisible();
  expect(requests).toHaveLength(requestsAfterReset);
  expect(requests).not.toContain('unfiltered');
});
