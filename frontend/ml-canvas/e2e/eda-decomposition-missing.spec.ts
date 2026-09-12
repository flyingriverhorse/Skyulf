import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(viewport.name, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('drills into missing and literal Unknown groups independently by keyboard', async ({ page }) => {
      /** The real browser request must retain JSON null through split and parent-selection refresh. */
      await mockBackend(page);
      await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
        id: '192', source_id: 'missing-groups', name: 'Missing groups', type: 'file', format: 'csv',
        rows: 4, columns: 2, created_at: '2026-09-12T10:00:00Z',
      }] } }));
      await page.route('**/api/eda/192/history', route => route.fulfill({ json: [] }));
      await page.route('**/api/eda/192/latest', route => route.fulfill({ json: {
        id: 192, status: 'COMPLETED', created_at: '2026-09-12T10:00:00Z',
        profile_data: {
          row_count: 4, column_count: 2, duplicate_rows: 0,
          missing_cells_percentage: 25, memory_usage_mb: 0.01, alerts: [],
          columns: Object.fromEntries(['group', 'detail'].map(name => [name, {
            name, dtype: 'Categorical', missing_count: name === 'group' ? 2 : 0,
            missing_percentage: name === 'group' ? 50 : 0,
            categorical_stats: { unique_count: 2, top_k: [], rare_labels_count: 0 },
          }])),
        },
      } }));
      const requests: Array<{ column: string; operator: string; value: string | null }> = [];
      await page.route('**/api/eda/192/decomposition', route => {
        const body = route.request().postDataJSON();
        if (!body.split_col) return route.fulfill({ json: [{ name: 'Total', value: 4, ratio: 1 }] });
        if (body.split_col === 'group') return route.fulfill({ json: [
          { name: 'Unknown', filter_value: null, value: 2, ratio: 0.5 },
          { name: 'Unknown', filter_value: 'Unknown', value: 1, ratio: 0.25 },
          { name: 'a', filter_value: 'a', value: 1, ratio: 0.25 },
        ] });
        requests.push(...body.filters);
        if (body.filters[0]?.value === null) return route.fulfill({ json: [
          { name: 'missing detail', filter_value: 'missing detail', value: 2, ratio: 1 },
        ] });
        if (body.filters[0]?.value === 'Unknown') return route.fulfill({ json: [
          { name: 'literal detail', filter_value: 'literal detail', value: 1, ratio: 1 },
        ] });
        return route.fulfill({ status: 400, json: { detail: 'Unexpected decomposition filter' } });
      });
      await page.goto('/eda?dataset_id=192');
      if (viewport.name === 'mobile') {
        await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
      }
      await page.getByRole('button', { name: 'Decomposition', exact: true }).click();
      await page.getByRole('button', { name: 'Total 4 (100%)' }).click();
      await page.getByTitle('Split further').click();
      await page.getByRole('button', { name: 'group', exact: true }).click();
      const missing = page.getByRole('button', { name: 'Unknown (missing) 2 (50%)' });
      const literal = page.getByRole('button', { name: 'Unknown 1 (25%)' });
      await missing.focus();
      await missing.press('Enter');
      await expect(missing).toHaveAttribute('aria-pressed', 'true');
      await expect(literal).toHaveAttribute('aria-pressed', 'false');
      expect(await missing.getAttribute('id')).not.toBe(await literal.getAttribute('id'));
      await page.getByTitle('Split further').click();
      await page.getByRole('button', { name: 'detail', exact: true }).click();
      await expect(page.getByText('missing detail', { exact: true })).toBeVisible();
      await literal.focus();
      await literal.press('Enter');
      await expect(page.getByText('literal detail', { exact: true })).toBeVisible();
      await expect(missing).toHaveAttribute('aria-pressed', 'false');
      await expect(literal).toHaveAttribute('aria-pressed', 'true');
      expect(requests).toEqual([
        { column: 'group', operator: '==', value: null },
        { column: 'group', operator: '==', value: 'Unknown' },
      ]);
    });
  });
}
