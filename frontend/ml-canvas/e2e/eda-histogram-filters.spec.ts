import { readFileSync } from 'node:fs';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/eda-histogram-boundaries.json', import.meta.url), 'utf8'));

for (const caseIndex of [0, 1, 2]) {
  test(`numeric histogram ${['first', 'middle', 'last'][caseIndex]} bin submits its exact range`, async ({ page }) => {
    // Python integration tests pin these histogram counts and filters to real analyzer output.
    const selected = fixture.cases[caseIndex];
    await mockBackend(page);
    await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
      id: '8080', source_id: 'histogram', name: 'Histogram edges', type: 'file', format: 'csv',
      rows: 67, columns: 2, created_at: '2026-09-21T00:00:00Z',
    }] } }));
    await page.route('**/api/eda/8080/history', route => route.fulfill({ json: [] }));
    await page.route('**/api/eda/8080/latest', route => route.fulfill({ json: {
      id: 8080, status: 'COMPLETED', config: { filters: fixture.existing_filters },
      profile_data: fixture.profile,
    } }));
    let submitted: unknown;
    await page.route('**/api/eda/8080/analyze', route => {
      submitted = route.request().postDataJSON().filters;
      return route.fulfill({ json: { id: 8081, status: 'PENDING' } });
    });
    await page.goto('/eda?dataset_id=8080');
    await page.getByRole('button', { name: 'Variables', exact: true }).click();
    await page.getByRole('button', { name: 'Expand All', exact: true }).click();
    const bars = page.locator('.recharts-bar-rectangle path');
    await expect(bars).toHaveCount(20);
    await bars.nth(selected.bin_index).hover();
    await expect(page.getByText(`Count: ${selected.expected_values.length}`, { exact: true })).toBeVisible();
    await bars.nth(selected.bin_index).click();
    await expect(page.getByRole('button', { name: 'Draft Filters (3)' })).toBeVisible();
    await expect(page.getByText('Report uses 1 applied filter until you apply draft changes.', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Apply filters', exact: true }).click();
    await expect.poll(() => submitted).toEqual(selected.filters);
  });
}
