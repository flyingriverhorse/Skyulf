import { expect, test } from '@playwright/test';
import type { EDAProfile } from '../src/core/types/edaProfile';
import { mockBackend } from './fixtures/mockApi';

declare global {
  interface Window {
    edaExportText: string[];
  }
}

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'mobile', width: 390, height: 844 },
]) {
  test.describe(`EDA target chart contracts on ${viewport.name}`, () => {
    test.use({ viewport: { width: viewport.width, height: viewport.height } });

    test('renders typed box plot details and exports only available ANOVA results', async ({ page }) => {
      /** Missing ANOVA results stay absent from exported images while true zero remains meaningful. */
      const profile: EDAProfile = {
        row_count: 4, column_count: 2, target_col: 'group',
        columns: {
          group: { name: 'group', dtype: 'Categorical', missing_count: 0, missing_percentage: 0 },
          amount: { name: 'amount', dtype: 'Numeric', missing_count: 0, missing_percentage: 0 },
        },
        target_correlations: { amount: null },
        target_interactions: [null, 0].map((p_value, index) => ({
          feature: `amount_${index}`, plot_type: 'boxplot', p_value,
          data: [{ name: 'group_a', stats: { min: 1, q1: 2, median: 3, q3: 4, max: index === 0 ? null : 5 } }],
        })),
      };
      await mockBackend(page);
      await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
        id: '194', source_id: 'target-export', name: 'Target export', type: 'file', format: 'csv',
        rows: 4, columns: 2, created_at: '2026-09-13T10:00:00Z',
      }] } }));
      await page.route('**/api/eda/194/history', route => route.fulfill({ json: [] }));
      await page.route('**/api/eda/194/latest', route => route.fulfill({ json: {
        id: 194, status: 'COMPLETED', profile_data: profile,
      } }));
      await page.addInitScript(() => {
        window.edaExportText = [];
        const original = CanvasRenderingContext2D.prototype.fillText;
        CanvasRenderingContext2D.prototype.fillText = function (...args) {
          window.edaExportText.push(args[0]);
          return original.apply(this, args);
        };
      });

      await page.goto('/eda?dataset_id=194');
      if (viewport.name === 'mobile') await page.getByRole('button', { name: 'Collapse Sidebar', exact: true }).click();
      await page.getByRole('button', { name: 'Target Analysis', exact: true }).click();

      const chart = page.locator('#interaction-chart-0');
      await chart.scrollIntoViewIfNeeded();
      await chart.locator('svg.recharts-surface').focus();
      await page.keyboard.press('ArrowRight');
      await expect(chart.getByText('Max: N/A', { exact: true })).toBeVisible();
      await expect(chart.getByText('Median: 3.00', { exact: true })).toBeVisible();
      await expect(chart.getByText('Min: 1.00', { exact: true })).toBeVisible();

      const downloadPromise = page.waitForEvent('download');
      await page.getByRole('button', { name: 'Download All Charts', exact: true }).click();
      const download = await downloadPromise;
      expect(download.suggestedFilename()).toBe('all-interactions.png');
      expect(await page.evaluate(() => window.edaExportText.filter(text => text.startsWith('ANOVA p:'))))
        .toEqual(['ANOVA p: 0.00e+0']);
    });
  });
}
