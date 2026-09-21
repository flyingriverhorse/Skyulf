import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';
import { readFile } from 'node:fs/promises';

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
      const sample = [-1, -0.4, -0.02, 0, 0.02, 0.4, 1, null];
      await mockProfile(page, Object.fromEntries(allColumns.map(name => [name, {
        name, dtype: 'Numeric', missing_count: 0, missing_percentage: 0,
      }])), {
        correlations: {
          columns: matrixColumns,
          values: matrixColumns.map(() => matrixColumns.map((_, column) => sample[column % sample.length])),
          total_columns: 25, omitted_columns: allColumns.slice(20),
        },
      });
      await openAnalysis(page, 'Correlations', viewport.name === 'mobile');
      const warning = page.getByText(/Correlation analysis used the first 20 of 25 numeric columns/);
      await expect(warning).toBeVisible();
      await expect(warning).toContainText('5 omitted: feature_20, feature_21, feature_22, feature_23, feature_24');
      await expect(warning).toContainText('These omitted columns are not included in the data table.');
      await expect(page.getByText(/Use the data table below for the full matrix/)).toHaveCount(0);
      const screenColors = await Promise.all(sample.map((value, index) =>
        page.getByTitle(`feature_0 vs feature_${index}: ${value === null ? 'N/A' : value.toFixed(3)}`, { exact: true })
          .evaluate(element => getComputedStyle(element).backgroundColor)));
      const downloaded = page.waitForEvent('download');
      await page.getByRole('button', { name: 'Download Matrix', exact: true }).click();
      const download = await downloaded;
      const png = await readFile((await download.path())!);
      const exportedColors = await page.evaluate(async ({ base64, columns }) => {
        const image = new Image();
        image.src = `data:image/png;base64,${base64}`;
        await image.decode();
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext('2d')!;
        context.drawImage(image, 0, 0);
        context.font = '12px sans-serif';
        const labelWidth = Math.max(...columns.map(name => context.measureText(name).width)) + 40;
        return Array.from({ length: 8 }, (_, index) => {
          const pixel = context.getImageData(labelWidth + index * 60 + 4, image.height - 50 - columns.length * 60 + 4, 1, 1).data;
          return `rgb(${pixel[0]}, ${pixel[1]}, ${pixel[2]})`;
        });
      }, { base64: png.toString('base64'), columns: matrixColumns });
      expect(exportedColors).toEqual(screenColors);
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
      let failRoot = true;
      let failSplit = true;
      await page.route('**/api/eda/193/decomposition', route => {
        const body = route.request().postDataJSON();
        if (!body.split_col && failRoot) {
          return route.fulfill({ status: 503, json: { detail: 'Root unavailable' } });
        }
        if (body.split_col === 'recorded_at' && failSplit) {
          return route.fulfill({ status: 503, json: { detail: 'Split unavailable' } });
        }
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
      await expect(page.getByRole('alert')).toContainText('Could not load decomposition');
      failRoot = false;
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
      await page.getByRole('button', { name: 'Total 4 (100%)', exact: true }).click();
      await page.getByTitle('Split further').click();
      await page.getByRole('button', { name: 'recorded_at', exact: true }).click();
      await expect(page.getByRole('alert')).toContainText('Could not split by recorded_at');
      failSplit = false;
      await expect(page.getByRole('button', { name: 'Total 4 (100%)', exact: true })).toBeVisible();
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
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
