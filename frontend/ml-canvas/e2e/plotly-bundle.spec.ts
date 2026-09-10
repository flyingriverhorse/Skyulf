import { readFile } from 'node:fs/promises';
import { expect, test } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

// Permit software WebGL in headless CI so the real 3D renderer is exercised.
test.use({ launchOptions: { args: ['--enable-unsafe-swiftshader'] } });

test('PCA renders a 3D projection and exports its PNG through the bundled Plotly instance', async ({ page }, testInfo) => {
  // A bundle or factory mismatch must fail real rendering/export, not just a mocked import.
  await mockBackend(page);
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: '101', source_id: 'projection', name: 'Projection data', type: 'file',
    format: 'csv', rows: 4, columns: 3, created_at: '2026-09-10T10:00:00Z',
  }] } }));
  await page.route('**/api/eda/101/history', route => route.fulfill({ json: [] }));
  await page.route('**/api/eda/101/latest', route => route.fulfill({ json: {
    id: 31, status: 'COMPLETED', profile_data: {
      row_count: 4, column_count: 3, columns: {}, alerts: [],
      pca_data: [
        { x: 1, y: 2, z: 3, label: 'A' },
        { x: 2, y: 4, z: 1, label: 'A' },
        { x: -1, y: -2, z: -3, label: 'B' },
        { x: -2, y: -4, z: -1, label: 'B' },
      ],
    },
  } }));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/eda?dataset_id=101');
  await page.getByRole('button', { name: 'PCA & Clusters', exact: true }).click();
  await page.getByRole('button', { name: 'Switch to 3D', exact: true }).click();

  const plot = page.locator('#pca-chart .js-plotly-plot');
  await expect(plot.locator('.gl-container canvas')).toBeVisible();
  await expect.poll(() => plot.evaluate(element => {
    const chart = element as HTMLElement & {
      data: { type: string; name: string; x: number[]; y: number[]; z: number[] }[];
    };
    return chart.data.map(({ type, name, x, y, z }) => ({ type, name, x, y, z }));
  })).toEqual([
    { type: 'scatter3d', name: 'A', x: [1, 2], y: [2, 4], z: [3, 1] },
    { type: 'scatter3d', name: 'B', x: [-1, -2], y: [-2, -4], z: [-3, -1] },
  ]);

  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download Chart', exact: true }).first().click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toBe('pca-analysis.png');
  const imagePath = testInfo.outputPath('pca-analysis.png');
  await download.saveAs(imagePath);
  const png = await readFile(imagePath);
  expect(png.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  expect(png.readUInt32BE(16)).toBe(1200);
  expect(png.readUInt32BE(20)).toBe(880);

  await page.getByRole('button', { name: 'Switch to 2D', exact: true }).click();
  await expect(plot).toHaveCount(0);
  await expect(page.locator('#pca-chart canvas')).toBeVisible();
});
