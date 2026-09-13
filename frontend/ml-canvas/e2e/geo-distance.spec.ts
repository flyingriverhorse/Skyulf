import { expect, test, type Page } from '@playwright/test';
import { mockBackend } from './fixtures/mockApi';

/** Capture the real converter output while the server response remains hermetic. */
async function previewGeoDistance(page: Page) {
  const pending = page.waitForRequest(request => request.method() === 'POST'
    && request.url().includes('/api/pipeline/preview?'));
  await page.getByRole('button', { name: 'Preview data', exact: true }).click();
  const request = await pending;
  const payload = request.postDataJSON() as {
    nodes: { node_id: string; step_type: string; params: Record<string, unknown> }[];
  };
  return payload.nodes.find(node => node.node_id === 'geo');
}

test('Geo Distance validates coordinates, previews both methods and keeps edits across panel widths', async ({ page }) => {
  // Exercise real keyboard validation, numeric selection and HTTP serialization on desktop and mobile.
  await mockBackend(page);
  await page.route('**/api/pipeline/registry', route => route.fulfill({ json: [] }));
  await page.route('**/api/pipeline/hyperparameters/*', route => route.fulfill({ json: [] }));
  await page.route('**/data/api/sources/usable', route => route.fulfill({ json: { sources: [{
    id: 'locations', name: 'Locations', rows: 2, columns: 5, file_type: 'csv',
  }] } }));
  await page.route('**/api/pipeline/datasets/*/schema', route => route.fulfill({ json: {
    columns: {
      lat: { name: 'lat', dtype: 'Float64' }, lon: { name: 'lon', dtype: 'Float64' },
      destination_lat: { name: 'destination_lat', dtype: 'Float64' },
      destination_lon: { name: 'destination_lon', dtype: 'Float64' },
      city: { name: 'city', dtype: 'String' },
    }, row_count: 2,
  } }));
  await page.route('**/api/pipeline/preview?*', route => route.fulfill({ json: {
    pipeline_id: 'geo-canvas', status: 'success', node_results: {}, recommendations: [],
    preview_data: [{ lat: 51.5, lon: -0.1, destination_lat: 51.6, destination_lon: -0.2, geo_distance_km: 13.094 }],
  } }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/canvas');
  await page.waitForFunction(() => '__skyulfTest' in window);
  await page.evaluate(() => {
    const state = (window as unknown as {
      __skyulfTest: { graphStore: { getState: () => Record<string, unknown> } };
    }).__skyulfTest.graphStore.getState();
    (state.setGraph as (nodes: unknown[], edges: unknown[]) => void)([
      {
        id: 'dataset', type: 'custom', position: { x: 0, y: 100 },
        data: { definitionType: 'dataset_node', datasetId: 'locations', datasetName: 'Locations' },
      },
      {
        id: 'geo', type: 'custom', position: { x: 350, y: 100 }, selected: true,
        data: { definitionType: 'GeoDistance', label: 'Geo Distance',
          lat1_col: '', lon1_col: '', lat2_col: '', lon2_col: '',
          method: 'haversine', unit: 'km', output_column: '',
        },
      },
    ], [{ id: 'edge', source: 'dataset', target: 'geo', sourceHandle: 'data', targetHandle: 'in' }]);
  });

  await page.getByRole('button', { name: /Configuration Geo Distance.*point 1 latitude/i }).press('Enter');
  const latitude = page.getByRole('combobox', { name: 'Point 1 latitude', exact: true });
  await expect(latitude).toBeFocused();
  await expect(latitude).toHaveAttribute('aria-invalid', 'true');
  await expect(latitude.getByRole('option', { name: 'city', exact: true })).toHaveCount(0);
  await latitude.press('Home');
  await latitude.press('ArrowDown');
  await expect(latitude).toHaveValue('lat');
  await expect(latitude).not.toHaveAttribute('aria-invalid');
  await page.getByRole('combobox', { name: 'Point 1 longitude', exact: true }).selectOption('lon');
  await page.getByRole('combobox', { name: 'Point 2 latitude', exact: true }).selectOption('destination_lat');
  await page.getByRole('combobox', { name: 'Point 2 longitude', exact: true }).selectOption('destination_lon');
  const coordinates = { lat1_col: 'lat', lon1_col: 'lon', lat2_col: 'destination_lat', lon2_col: 'destination_lon' };
  expect(await previewGeoDistance(page)).toMatchObject({
    step_type: 'GeoDistance', params: { ...coordinates, method: 'haversine', unit: 'km', output_column: '' },
  });
  await expect(page.getByRole('columnheader', { name: 'geo_distance_km', exact: true })).toBeVisible();

  await page.getByRole('button', { name: 'Expand settings panel', exact: true }).click();
  await expect(latitude).toHaveValue('lat');
  await expect.poll(async () => {
    const left = await latitude.boundingBox();
    const right = await page.getByRole('combobox', { name: 'Distance method', exact: true }).boundingBox();
    return left && right ? Math.abs(left.y - right.y) : Number.POSITIVE_INFINITY;
  }).toBeLessThan(2);
  await page.getByRole('combobox', { name: 'Distance method', exact: true }).selectOption('euclidean');
  await expect(page.getByText(/Flat-earth approximation for nearby points/)).toBeVisible();
  const unit = page.getByRole('combobox', { name: 'Distance unit', exact: true });
  await unit.press('End');
  await expect(unit).toHaveValue('mi');
  const output = page.getByRole('textbox', { name: 'Output column (optional)', exact: true });
  await expect(output).toHaveAttribute('placeholder', 'geo_distance_mi');
  await output.fill('journey_distance');
  await page.screenshot({ path: 'test-results/geo-distance-expanded.png', animations: 'disabled' });
  await page.getByRole('button', { name: 'Collapse settings panel', exact: true }).click();
  await expect(output).toHaveValue('journey_distance');
  expect(await previewGeoDistance(page)).toMatchObject({
    step_type: 'GeoDistance', params: { ...coordinates, method: 'euclidean', unit: 'mi', output_column: 'journey_distance' },
  });
  const settings = page.getByRole('complementary', { name: 'Node settings', exact: true });
  expect(await settings.evaluate(element => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: 'test-results/geo-distance-compact.png', animations: 'disabled' });

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole('button', { name: 'Read-only', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(latitude).toHaveCount(0);
  await page.screenshot({ path: 'test-results/geo-distance-mobile.png', animations: 'disabled' });
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(latitude).toHaveValue('lat');
  await expect(unit).toHaveValue('mi');
  await expect(output).toHaveValue('journey_distance');
  await output.fill('');
  await expect(output).toHaveAttribute('placeholder', 'geo_distance_mi');
  await expect(page.getByRole('button', { name: /Configuration Geo Distance/i })).toHaveCount(0);
});
